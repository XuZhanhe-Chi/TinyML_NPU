# -*- coding: utf-8 -*-
"""
Graph partitioning for mixed NPU/CPU execution (plan pipeline).

This pass walks the IR in (assumed) topological order and partitions it into
execution steps:
  - NPU step: a maximal consecutive run of NPU-supported ops.
  - CPU step: a single unsupported-but-fallback op (e.g. Add / ConcatC).
  - ALIAS step: a single view op (e.g. Reshape / Identity / Flatten) that
    does not move data.

The output of this pass is a lightweight StepSpec list describing step
boundaries and step inputs/outputs in terms of IR tensor names. Later stages
map StepSpec into the runtime Plan ABI (tensor IDs and step descriptors).

Strictness:
  - Any op that is neither NPU-supported nor a known CPU/ALIAS fallback raises
    ValueError with op name/op_type.
  - The pass does not attempt to silently ignore unsupported ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from venuscore_compiler.ir.ops import (
    VcAdd,
    VcAvgPool,
    VcConcatC,
    VcConv2D,
    VcDepthwiseConv,
    VcFlatten,
    VcFullyConnected,
    VcIdentity,
    VcMaxPool,
    VcOp,
    VcPointwiseConv,
    VcReshape,
)
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.plan.types import CpuKernel, StepType

__all__ = ["StepSpec", "partition_program"]


@dataclass(frozen=True)
class StepSpec:
    """
    Compiler-side step specification (pre-ABI).

    Attributes:
        step_type:
            NPU / CPU / ALIAS.
        op_names:
            Names of ops included in this step (one or many for NPU).
        inputs:
            IR tensor names used as inputs by this step (activation tensors).
        outputs:
            IR tensor names produced by this step that are visible outside.
        cpu_kernel:
            For CPU steps, which built-in kernel to use.
        axis:
            For CONCAT_C, axis is fixed to C (1 in NCHW).
    """

    step_type: StepType
    op_names: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    cpu_kernel: CpuKernel | None = None
    axis: int | None = None
    activation: str | None = None


_NPU_OP_TYPES: tuple[type[VcOp], ...] = (
    VcConv2D,
    VcPointwiseConv,
    VcDepthwiseConv,
    VcAvgPool,
    VcMaxPool,
    VcFullyConnected,
)
_ALIAS_OP_TYPES: tuple[type[VcOp], ...] = (VcIdentity, VcReshape, VcFlatten)
_CPU_OP_TYPES: tuple[type[VcOp], ...] = (VcAdd, VcConcatC)


def partition_program(program: VcProgram) -> List[StepSpec]:
    """
    Partition a VcProgram into mixed execution steps.

    The program ops are assumed to be in topological order.
    """
    ops = list(program.ops)
    if not ops:
        return []

    storage_alias = _collect_storage_alias(program, ops)

    def resolve_storage(name: str) -> str:
        return _resolve_alias_root(name, storage_alias)

    producer: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    # Track producers on the storage root so alias names (e.g. Q/DQ outputs)
    # are treated as internal edges during step boundary discovery.
    producer_root: Dict[str, int] = {}
    for idx, op in enumerate(ops):
        for t in op.outputs:
            if t in producer:
                raise ValueError(f"[partition] Tensor '{t}' has multiple producers: {producer[t]} and {idx}")
            producer[t] = idx
            if not isinstance(op, _ALIAS_OP_TYPES):
                r = resolve_storage(t)
                if r in producer_root and producer_root[r] != idx:
                    raise ValueError(
                        f"[partition] Storage root '{r}' has multiple producers: {producer_root[r]} and {idx}"
                    )
                producer_root[r] = idx
        for t in op.inputs:
            consumers.setdefault(t, []).append(idx)

    # Propagate consumers to storage roots so that a producer output is not
    # mistaken as a graph output when only its aliases are consumed.
    consumers_root: Dict[str, List[int]] = {}
    for t, idxs in consumers.items():
        r = resolve_storage(t)
        dst = consumers_root.setdefault(r, [])
        for i in idxs:
            if i not in dst:
                dst.append(i)

    # Some alias ops represent user-visible graph outputs even though they do
    # not change the underlying storage root (e.g. Flatten producing "logits").
    # Track these by *name*, not by root, so Plan outputs can reference them.
    alias_produced: Set[str] = set()
    for op in ops:
        if isinstance(op, _ALIAS_OP_TYPES):
            alias_produced.update(op.outputs)
    graph_output_alias_names: Set[str] = {t for t in alias_produced if t not in consumers}

    graph_output_roots: Set[str] = set()
    for t_name in producer.keys():
        r = resolve_storage(t_name)
        if r not in consumers_root:
            graph_output_roots.add(r)

    step_specs: List[StepSpec] = []

    current_npu_ops: List[int] = []

    def _flush_npu() -> None:
        nonlocal current_npu_ops
        if not current_npu_ops:
            return
        spec = _build_step_spec(
            program,
            current_npu_ops,
            producer=producer,
            producer_root=producer_root,
            consumers=consumers,
            consumers_root=consumers_root,
            graph_output_roots=graph_output_roots,
            graph_output_alias_names=graph_output_alias_names,
            resolve_storage=resolve_storage,
        )
        step_specs.append(spec)
        current_npu_ops = []

    for idx, op in enumerate(ops):
        if isinstance(op, _NPU_OP_TYPES):
            current_npu_ops.append(idx)
            continue

        # Non-NPU op breaks the current NPU run.
        _flush_npu()

        if isinstance(op, _ALIAS_OP_TYPES):
            spec = _build_step_spec(
                program,
                [idx],
                producer=producer,
                producer_root=producer_root,
                consumers=consumers,
                consumers_root=consumers_root,
                graph_output_roots=graph_output_roots,
                graph_output_alias_names=graph_output_alias_names,
                resolve_storage=resolve_storage,
            )
            step_specs.append(spec)
            continue

        if isinstance(op, _CPU_OP_TYPES):
            spec = _build_step_spec(
                program,
                [idx],
                producer=producer,
                producer_root=producer_root,
                consumers=consumers,
                consumers_root=consumers_root,
                graph_output_roots=graph_output_roots,
                graph_output_alias_names=graph_output_alias_names,
                resolve_storage=resolve_storage,
            )
            step_specs.append(spec)
            continue

        raise ValueError(f"[partition] Unsupported op for plan fallback: '{op.name}' (op_type={op.op_type})")

    _flush_npu()

    return step_specs


def _build_step_spec(
    program: VcProgram,
    op_indices: Sequence[int],
    *,
    producer: Dict[str, int],
    producer_root: Dict[str, int],
    consumers: Dict[str, List[int]],
    consumers_root: Dict[str, List[int]],
    graph_output_roots: Set[str],
    graph_output_alias_names: Set[str],
    resolve_storage,
) -> StepSpec:
    ops = program.ops
    idx_set = set(op_indices)
    ops_in_step = [ops[i] for i in op_indices]

    # Determine step type and (optional) CPU kernel.
    step_type = _classify_step_type(ops_in_step)
    cpu_kernel: CpuKernel | None = None
    axis: int | None = None
    activation: str | None = None
    if step_type == StepType.CPU:
        op0 = ops_in_step[0]
        if isinstance(op0, VcAdd):
            cpu_kernel = CpuKernel.ADD
            activation = getattr(op0, "activation", None)
        elif isinstance(op0, VcConcatC):
            cpu_kernel = CpuKernel.CONCAT_C
            axis = op0.axis
            activation = getattr(op0, "activation", None)
        else:
            raise ValueError(f"[partition] Unknown CPU fallback op: {type(op0).__name__}")

    # Step inputs: any op input not produced inside the step.
    produced: Set[str] = set()
    produced_roots: Set[str] = set()
    produced_in_order: List[str] = []
    for op in ops_in_step:
        for o in op.outputs:
            produced.add(o)
            produced_in_order.append(o)
            produced_roots.add(resolve_storage(o))

    inputs: List[str] = []
    for op in ops_in_step:
        for t in op.inputs:
            root = resolve_storage(t)
            if root in produced_roots and producer_root.get(root, -1) in idx_set:
                continue
            if t not in inputs:
                inputs.append(t)

    # Step outputs: pick one representative name per visible storage root.
    visible_roots: Set[str] = set()
    for o in produced_in_order:
        r = resolve_storage(o)
        if o in graph_output_alias_names:
            visible_roots.add(r)
            continue
        if r in graph_output_roots:
            visible_roots.add(r)
            continue
        for c in consumers_root.get(r, []):
            if c not in idx_set:
                visible_roots.add(r)
                break

    outputs: List[str] = []
    seen_roots: Set[str] = set()
    for op in ops_in_step:
        for o in op.outputs:
            r = resolve_storage(o)
            if r in visible_roots and r not in seen_roots:
                outputs.append(o)
                seen_roots.add(r)

    # Preserve a stable ordering: use program op order for outputs.
    outputs = _stable_output_order(outputs, ops_in_step)

    # Basic sanity: enforce v1 ABI limits early.
    if len(inputs) > 4:
        raise ValueError(f"[partition] Step has too many inputs ({len(inputs)}): ops={[_o.name for _o in ops_in_step]}")
    if len(outputs) > 2:
        raise ValueError(f"[partition] Step has too many outputs ({len(outputs)}): ops={[_o.name for _o in ops_in_step]}")

    return StepSpec(
        step_type=step_type,
        op_names=tuple(op.name for op in ops_in_step),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        cpu_kernel=cpu_kernel,
        axis=axis,
        activation=activation,
    )


def _classify_step_type(ops_in_step: Sequence[VcOp]) -> StepType:
    if not ops_in_step:
        raise ValueError("[partition] Empty step")
    op0 = ops_in_step[0]
    if len(ops_in_step) > 1:
        # Only NPU steps are allowed to contain multiple ops in v1.
        if not isinstance(op0, _NPU_OP_TYPES):
            raise ValueError(f"[partition] Non-NPU step cannot contain multiple ops: {type(op0).__name__}")
        return StepType.NPU

    if isinstance(op0, _NPU_OP_TYPES):
        return StepType.NPU
    if isinstance(op0, _ALIAS_OP_TYPES):
        return StepType.ALIAS
    if isinstance(op0, _CPU_OP_TYPES):
        return StepType.CPU
    raise ValueError(f"[partition] Unknown op category: {type(op0).__name__}")


def _stable_output_order(outputs: List[str], ops_in_step: Sequence[VcOp]) -> List[str]:
    """Try to keep output ordering deterministic, preferring later op outputs."""
    if not outputs:
        return []
    out_set = set(outputs)
    ordered: List[str] = []
    for op in ops_in_step:
        for t in op.outputs:
            if t in out_set and t not in ordered:
                ordered.append(t)
    # Fallback for anything missing.
    for t in outputs:
        if t not in ordered:
            ordered.append(t)
    return ordered


def _collect_storage_alias(program: VcProgram, ops: Sequence[VcOp]) -> Dict[str, str]:
    """
    Collect a best-effort storage alias map for step boundary discovery.

    The ONNX frontend may populate program.metadata["storage_alias"] to model
    no-op view nodes (e.g. Q/DQ, Identity, Reshape). We also treat explicit
    alias IR ops (VcIdentity/VcReshape/VcFlatten) as storage aliases.
    """
    storage_alias: Dict[str, str] = {}

    meta = getattr(program, "metadata", {}) or {}
    if isinstance(meta, dict):
        sa = meta.get("storage_alias", {})
        if isinstance(sa, dict):
            for k, v in sa.items():
                storage_alias[str(k)] = str(v)

    for op in ops:
        if isinstance(op, _ALIAS_OP_TYPES) and op.inputs and op.outputs:
            storage_alias[str(op.outputs[0])] = str(op.inputs[0])

    return storage_alias


def _resolve_alias_root(name: str, storage_alias: Dict[str, str]) -> str:
    """Follow storage_alias chain to its root. Detect cycles."""
    cur = name
    seen: Set[str] = set()
    while cur in storage_alias:
        if cur in seen:
            chain = " -> ".join(list(seen) + [cur])
            raise ValueError(f"[partition] storage_alias cycle detected at '{cur}' (start='{name}'): {chain}")
        seen.add(cur)
        cur = storage_alias[cur]
    return cur
