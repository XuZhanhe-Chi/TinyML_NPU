# -*- coding: utf-8 -*-
"""
Build a mixed-execution Plan (NPU/CPU/ALIAS) from IR + compiled NPU artifacts.

This module glues together:
  - midend.partition_fallback: step partitioning at the IR level
  - backend tiling/codegen: NPU uOP/Param generation (already implemented)
  - plan.types: runtime ABI-friendly descriptors (tensor offsets + step list)

The resulting Plan is meant to be embedded into bundle.h and/or serialized into
metadata so firmware can execute a network as a sequence of NPU/CPU steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from venuscore_compiler.ir.ops import VcAdd, VcConcatC, VcFlatten, VcIdentity, VcReshape
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.ir.tensor import VcTensor
from venuscore_compiler.midend.partition_fallback import StepSpec, partition_program
from venuscore_compiler.midend.types import TilePlan
from venuscore_compiler.backend.memory_planner import MemoryPlan, _estimate_param_block_size
from venuscore_compiler.plan.types import (
    AliasStepDesc,
    CpuActivation,
    CpuKernel,
    CpuStepDesc,
    NpuStepDesc,
    Plan,
    TensorDesc,
)

UOP_WORDS_PER_UOP = 8  # 32B / 4B
WORD_BYTES = 4
TENSOR_ALIGNMENT_BYTES = 16

__all__ = ["build_plan", "StepSpec"]


@dataclass(frozen=True)
class _TensorInfo:
    name: str
    shape_nchw: tuple[int, int, int, int]
    size_bytes: int
    offset_bytes: int


def build_plan(
    program: VcProgram,
    tile_plan: TilePlan,
    memory_plan: MemoryPlan,
    uops_len_bytes: int,
    params_len_bytes: int,
    *,
    step_specs: Optional[Sequence[StepSpec]] = None,
    emit_alias_steps: bool = False,
) -> Plan:
    """
    Build a runtime Plan for mixed NPU/CPU execution.

    Args:
        program:
            Full IR program (may contain CPU fallback ops such as VcAdd/VcConcatC).
        tile_plan:
            Midend TilePlan for NPU-supported ops.
        memory_plan:
            Backend MemoryPlan for NPU tiles (used to map uops/params subranges).
        uops_len_bytes:
            Total length of uops.bin in bytes.
        params_len_bytes:
            Total length of params.bin in bytes.
        step_specs:
            Optional precomputed StepSpec list. If None, partition_program(program) is used.

    Returns:
        A populated Plan instance.
    """
    if step_specs is None:
        step_specs = partition_program(program)

    # v1 runtime does not require explicit ALIAS steps (reshape/identity are
    # represented as shared storage offsets in TensorDesc). Keeping them can be
    # useful for debugging, but the default is to omit them to simplify the step
    # list (Plan is allowed to have 0 ALIAS steps).
    if not emit_alias_steps:
        step_specs = [s for s in step_specs if s.step_type.name != "ALIAS"]

    tensor_infos, arena_bytes = _plan_activation_arena(program, memory_plan)
    plan = Plan(
        arena_bytes=arena_bytes,
        uops_len_words=_ceil_div(uops_len_bytes, WORD_BYTES),
        params_len_words=_ceil_div(params_len_bytes, WORD_BYTES),
    )

    # Assign stable tensor IDs: 0..N-1 in sorted name order.
    for tid, name in enumerate(sorted(tensor_infos.keys())):
        info = tensor_infos[name]
        quant_index = None
        t = program.tensors.get(name)
        if t is not None:
            q_scheme = getattr(t, "q_scheme", "none")
            scale = getattr(t, "scale", None)
            if q_scheme == "symmetric_per_tensor" and isinstance(scale, (int, float)):
                quant_index = plan.get_or_add_quant_scale(float(scale))
        desc = TensorDesc(
            tensor_id=tid,
            name=name,
            offset_bytes=info.offset_bytes,
            size_bytes=info.size_bytes,
            shape=info.shape_nchw,
            quant_index=quant_index,
        )
        plan.add_tensor(desc)

    # Step descriptors
    npu_ranges = _build_npu_op_ranges(tile_plan)
    param_ranges = _build_param_ranges(tile_plan, memory_plan)

    for spec in step_specs:
        in_ids = tuple(plan.get_tensor_id(n) for n in spec.inputs)
        out_ids = tuple(plan.get_tensor_id(n) for n in spec.outputs)

        if spec.step_type.name == "NPU":
            uop_off_words, uop_words = _npu_uop_range_for_step(spec, npu_ranges)
            param_off_words, param_words = _npu_param_range_for_step(spec, param_ranges)
            plan.steps.append(
                NpuStepDesc(
                    inputs=in_ids,
                    outputs=out_ids,
                    uop_off_words=uop_off_words,
                    uop_words=uop_words,
                    param_off_words=param_off_words,
                    param_words=param_words,
                )
            )
        elif spec.step_type.name == "CPU":
            if spec.cpu_kernel is None:
                raise ValueError(f"[plan] CPU step missing cpu_kernel: ops={spec.op_names}")
            plan.steps.append(
                CpuStepDesc(
                    inputs=in_ids,
                    outputs=out_ids,
                    kernel=spec.cpu_kernel,
                    activation=_map_cpu_activation(spec.activation),
                    axis=int(spec.axis or 1),
                )
            )
            _validate_cpu_step_quant(program, spec)
            _validate_cpu_step_shapes(program, spec)
        elif spec.step_type.name == "ALIAS":
            # Explicit ALIAS steps are optional. When enabled, they model
            # reshape/identity view changes as a no-op step.
            plan.steps.append(AliasStepDesc(inputs=in_ids, outputs=out_ids))
            _validate_alias_step_shapes(program, spec)
        else:
            raise ValueError(f"[plan] Unknown StepType: {spec.step_type}")

    return plan


# -----------------------------------------------------------------------------
# Tensor arena planning
# -----------------------------------------------------------------------------


def _plan_activation_arena(program: VcProgram, memory_plan: MemoryPlan) -> tuple[Dict[str, _TensorInfo], int]:
    """
    Assign activation tensor offsets in a single arena (v1: linear allocation).

    - All activation tensors referenced by op.inputs/op.outputs are included.
    - Parameter tensors referenced by op.weight/op.bias are excluded.
    - Alias ops (Identity/Reshape/Flatten) reuse the input tensor's storage.
    - storage_alias from program.metadata (ONNX frontend) is honored as well.

    Important:
      - Offsets for tensors that already have a concrete address in MemoryPlan are
        preserved so that CPU steps read/write the same buffers as NPU uOPs.
      - Any remaining activation tensors (typically produced/consumed by CPU-only
        ops) are appended after the existing activation region.
    """
    param_names = _collect_param_tensor_names(program.ops)
    activation_names = _collect_activation_tensor_names(program.ops, param_names)

    storage_alias: Dict[str, str] = {}
    meta = getattr(program, "metadata", {}) or {}
    if isinstance(meta, dict):
        sa = meta.get("storage_alias", {})
        if isinstance(sa, dict):
            storage_alias.update({str(k): str(v) for k, v in sa.items()})

    # Alias ops in IR should also be treated as storage aliases.
    for op in program.ops:
        if isinstance(op, (VcIdentity, VcReshape, VcFlatten)):
            if op.inputs and op.outputs:
                storage_alias[op.outputs[0]] = op.inputs[0]

    # Build NCHW shapes and size bytes for all activation tensors.
    tensor_nchw: Dict[str, tuple[int, int, int, int]] = {}
    tensor_size: Dict[str, int] = {}
    for name in activation_names:
        t = program.tensors.get(name)
        if t is None:
            raise ValueError(f"[plan] Missing tensor '{name}' referenced by ops.")
        shape_nchw = _tensor_shape_as_nchw(t)
        tensor_nchw[name] = shape_nchw
        tensor_size[name] = _activation_size_bytes(shape_nchw)

    # Validate alias size compatibility (must match in bytes under NCHWc4).
    for out_name, in_name in list(storage_alias.items()):
        if out_name not in activation_names:
            continue
        if in_name not in activation_names:
            # Allow alias to a graph input / upstream name not in set only if present in tensors.
            if in_name not in program.tensors:
                raise ValueError(f"[plan] storage_alias refers to unknown tensor '{in_name}' (for '{out_name}').")
            activation_names.add(in_name)
            t_in = program.tensors[in_name]
            tensor_nchw[in_name] = _tensor_shape_as_nchw(t_in)
            tensor_size[in_name] = _activation_size_bytes(tensor_nchw[in_name])
        if tensor_size.get(out_name) != tensor_size.get(in_name):
            raise ValueError(
                f"[plan] Alias size mismatch: '{out_name}' ({tensor_nchw[out_name]} => {tensor_size[out_name]}B) "
                f"aliases '{in_name}' ({tensor_nchw[in_name]} => {tensor_size[in_name]}B)."
            )

    # Allocate offsets for roots only, then map aliases to roots.
    root_of: Dict[str, str] = {}
    for name in activation_names:
        root_of[name] = _resolve_alias_root(name, storage_alias)

    roots = sorted(set(root_of.values()))
    offset_by_root: Dict[str, int] = {}

    # First, preserve any pre-planned activation addresses from MemoryPlan.
    for r in roots:
        off = _lookup_activation_offset(memory_plan, r)
        if off is not None:
            offset_by_root[r] = off

    # Then, append any remaining roots after the current activation footprint.
    cursor = 0
    for r, off in offset_by_root.items():
        cursor = max(cursor, int(off) + int(tensor_size.get(r, 0)))

    for r in roots:
        if r in offset_by_root:
            continue
        cursor = _align_up(cursor, TENSOR_ALIGNMENT_BYTES)
        offset_by_root[r] = cursor
        cursor += tensor_size.get(r, 0)

    tensor_infos: Dict[str, _TensorInfo] = {}
    for name in activation_names:
        r = root_of[name]
        off = offset_by_root[r]
        tensor_infos[name] = _TensorInfo(
            name=name,
            shape_nchw=tensor_nchw.get(name, (1, 1, 1, 1)),
            size_bytes=tensor_size.get(name, 0),
            offset_bytes=off,
        )
    # IDs assigned later in build_plan based on sorted(tensor_infos.keys()).
    arena_bytes = _align_up(cursor, TENSOR_ALIGNMENT_BYTES)
    return tensor_infos, arena_bytes


def _lookup_activation_offset(memory_plan: MemoryPlan, name: str) -> Optional[int]:
    """
    Return a pre-planned activation base address for a tensor name, if present.

    The current backend uses MemoryPlan.ifm_offsets/ofm_offsets as the single
    source of truth for NPU FI_ADDR/FO_ADDR base computation.
    """
    ifm = memory_plan.ifm_offsets.get(name)
    ofm = memory_plan.ofm_offsets.get(name)
    if ifm is not None and ofm is not None and int(ifm) != int(ofm):
        raise ValueError(
            f"[plan] Tensor '{name}' has conflicting activation bases in MemoryPlan: "
            f"ifm_offsets=0x{int(ifm):X}, ofm_offsets=0x{int(ofm):X}"
        )
    if ifm is not None:
        return int(ifm)
    if ofm is not None:
        return int(ofm)
    return None


def _collect_param_tensor_names(ops: Iterable[object]) -> set[str]:
    names: set[str] = set()
    for op in ops:
        w = getattr(op, "weight", None)
        b = getattr(op, "bias", None)
        if isinstance(w, str) and w:
            names.add(w)
        if isinstance(b, str) and b:
            names.add(b)
    return names


def _collect_activation_tensor_names(ops: Iterable[object], param_names: set[str]) -> set[str]:
    names: set[str] = set()
    for op in ops:
        for t in getattr(op, "inputs", []) or []:
            if isinstance(t, str) and t and t not in param_names:
                names.add(t)
        for t in getattr(op, "outputs", []) or []:
            if isinstance(t, str) and t and t not in param_names:
                names.add(t)
    return names


def _resolve_alias_root(name: str, storage_alias: Dict[str, str]) -> str:
    seen: set[str] = set()
    cur = name
    while cur in storage_alias:
        if cur in seen:
            raise ValueError(f"[plan] storage_alias cycle detected at '{cur}' (start='{name}').")
        seen.add(cur)
        cur = storage_alias[cur]
    return cur


def _tensor_shape_as_nchw(t: VcTensor) -> tuple[int, int, int, int]:
    shape = tuple(int(x) for x in t.shape)
    if len(shape) != 4:
        raise ValueError(f"[plan] Tensor '{t.name}' must be 4D, got shape={shape}.")
    layout = (getattr(t, "layout", "NCHW") or "NCHW").upper()
    n, d1, d2, d3 = shape
    if layout == "NCHW":
        return int(n), int(d1), int(d2), int(d3)
    if layout == "NHWC":
        return int(n), int(d3), int(d1), int(d2)
    raise ValueError(f"[plan] Unsupported tensor layout '{layout}' for tensor '{t.name}'.")


def _activation_size_bytes(shape_nchw: tuple[int, int, int, int]) -> int:
    n, c, h, w = shape_nchw
    if n != 1:
        raise ValueError(f"[plan] Only N==1 is supported in plan v1, got N={n}.")
    c4 = _ceil_div(c, 4)
    return int(n) * int(c4) * int(h) * int(w) * WORD_BYTES


# -----------------------------------------------------------------------------
# NPU step range helpers
# -----------------------------------------------------------------------------


def _build_npu_op_ranges(tile_plan: TilePlan) -> Dict[str, tuple[int, int]]:
    """
    Map op_name -> (uop_start_index, uop_count) in the global uop stream.
    """
    ranges: Dict[str, tuple[int, int]] = {}
    for op_name, tiles in tile_plan.tiles_by_op.items():
        if not tiles:
            continue
        start = min(int(t.tile_index) for t in tiles)
        count = len(tiles)
        ranges[op_name] = (start, count)
    return ranges


def _build_param_ranges(tile_plan: TilePlan, memory_plan: MemoryPlan) -> Dict[str, tuple[int, int]]:
    """
    Map op_name -> (param_off_bytes, param_len_bytes) covering all param blocks for that op.

    The range is relative to memory_plan.param_base.
    """
    ranges: Dict[str, tuple[int, int]] = {}
    for op_name, tiles in tile_plan.tiles_by_op.items():
        if not tiles:
            continue
        blocks: Dict[int, int] = {}  # off_bytes -> size_bytes
        for tile in tiles:
            addr = memory_plan.param_offsets.get((op_name, tile.tile_id))
            if addr is None:
                continue
            off = int(addr) - int(memory_plan.param_base)
            if off in blocks:
                continue
            cfg = memory_plan.layer_configs.get(tile.layer_id)
            blocks[off] = int(_estimate_param_block_size(tile, cfg))
        if not blocks:
            ranges[op_name] = (0, 0)
            continue
        start = min(blocks.keys())
        end = max(off + size for off, size in blocks.items())
        ranges[op_name] = (start, max(0, end - start))
    return ranges


def _npu_uop_range_for_step(spec: StepSpec, npu_ranges: Dict[str, tuple[int, int]]) -> tuple[int, int]:
    starts: List[int] = []
    ends: List[int] = []
    total = 0
    for op_name in spec.op_names:
        if op_name not in npu_ranges:
            raise ValueError(f"[plan] NPU step references unknown NPU op '{op_name}'.")
        s, cnt = npu_ranges[op_name]
        starts.append(s)
        ends.append(s + cnt)
        total += cnt
    start = min(starts) if starts else 0
    end = max(ends) if ends else start
    if (end - start) != total:
        raise ValueError(f"[plan] NPU uop range is not contiguous for ops={spec.op_names}.")
    return start * UOP_WORDS_PER_UOP, total * UOP_WORDS_PER_UOP


def _npu_param_range_for_step(spec: StepSpec, param_ranges: Dict[str, tuple[int, int]]) -> tuple[int, int]:
    starts: List[int] = []
    ends: List[int] = []
    for op_name in spec.op_names:
        if op_name not in param_ranges:
            # Ops without params are allowed (e.g. avgpool); treat as empty.
            continue
        off, ln = param_ranges[op_name]
        starts.append(off)
        ends.append(off + ln)
    if not starts:
        return 0, 0
    start = min(starts)
    end = max(ends)
    # Convert to words (must be word-aligned in v1).
    if (start % WORD_BYTES) != 0 or (end % WORD_BYTES) != 0:
        raise ValueError(f"[plan] Param range is not word-aligned: start={start}, end={end}")
    return start // WORD_BYTES, (end - start) // WORD_BYTES


# -----------------------------------------------------------------------------
# CPU/ALIAS validation (strict)
# -----------------------------------------------------------------------------


def _validate_cpu_step_shapes(program: VcProgram, spec: StepSpec) -> None:
    # Only v1 CPU kernels are supported here; enforce basic shape constraints.
    if spec.cpu_kernel == CpuKernel.ADD:
        if len(spec.inputs) != 2 or len(spec.outputs) != 1:
            raise ValueError(f"[plan] ADD expects 2 inputs and 1 output, got {spec.inputs} -> {spec.outputs}")
        a = _tensor_shape_as_nchw(program.tensors[spec.inputs[0]])
        b = _tensor_shape_as_nchw(program.tensors[spec.inputs[1]])
        y = _tensor_shape_as_nchw(program.tensors[spec.outputs[0]])
        if a != b or a != y:
            raise ValueError(f"[plan] ADD shape mismatch: a={a}, b={b}, y={y}")

    if spec.cpu_kernel == CpuKernel.CONCAT_C:
        if len(spec.inputs) < 2 or len(spec.outputs) != 1:
            raise ValueError(
                f"[plan] CONCAT_C expects >=2 inputs and 1 output, got {spec.inputs} -> {spec.outputs}"
            )
        out = _tensor_shape_as_nchw(program.tensors[spec.outputs[0]])
        n, c_out, h, w = out
        c_sum = 0
        for inp_name in spec.inputs:
            n_i, c_i, h_i, w_i = _tensor_shape_as_nchw(program.tensors[inp_name])
            if (n_i, h_i, w_i) != (n, h, w):
                raise ValueError(f"[plan] CONCAT_C spatial mismatch: {inp_name}={n_i,c_i,h_i,w_i}, out={out}")
            c_sum += c_i
        if c_sum != c_out:
            raise ValueError(f"[plan] CONCAT_C channel mismatch: sum_in={c_sum}, out_c={c_out}")


def _map_cpu_activation(act: str | None) -> CpuActivation:
    if act is None:
        return CpuActivation.NONE
    a = str(act).strip().lower()
    if a in ("", "none"):
        return CpuActivation.NONE
    if a == "relu":
        return CpuActivation.RELU
    if a == "relu6":
        return CpuActivation.RELU6
    raise ValueError(f"[plan] Unsupported CPU activation: {act!r}")


def _validate_cpu_step_quant(program: VcProgram, spec: StepSpec) -> None:
    """
    Enforce a strict v1 quantization contract for CPU fallback:
      - symmetric int8 (zero_point == 0 by design)
      - per-tensor symmetric scales (scalar) are required when present
      - ADD allows mismatched input scales (requant); CONCAT_C requires matching scales
    """
    # If IR has no quant info, allow (treat as raw int8).
    def _get_scale(t: VcTensor) -> Optional[float]:
        if getattr(t, "q_scheme", "none") == "none":
            return None
        if getattr(t, "q_scheme", "") != "symmetric_per_tensor":
            raise ValueError(
                f"[plan] CPU fallback requires symmetric_per_tensor quant for now, got q_scheme={t.q_scheme!r} "
                f"for tensor '{t.name}'."
            )
        s = getattr(t, "scale", None)
        if s is None:
            return None
        if isinstance(s, (int, float)):
            return float(s)
        raise ValueError(f"[plan] Expected scalar scale for tensor '{t.name}', got {type(s).__name__}")

    scales: List[Optional[float]] = []
    for n in list(spec.inputs) + list(spec.outputs):
        t = program.tensors.get(n)
        if t is None:
            continue
        scales.append(_get_scale(t))

    # If any scale is present, require all present.
    present = [s for s in scales if s is not None]
    if not present:
        return
    if len(present) != len(scales):
        raise ValueError("[plan] CPU fallback requires all tensors in the step to have quant scales if any has.")

    # For CONCAT_C, scales must match because it's a pure byte copy of NCHWc4 planes.
    if spec.cpu_kernel == CpuKernel.CONCAT_C:
        ref = float(present[0])
        for s in present[1:]:
            if abs(float(s) - ref) > 1e-6:
                raise ValueError(f"[plan] CONCAT_C requires matching scales, got {present}")

    # For ADD, allow mismatched scales (handled by scale-aware CPU add kernel).


def _validate_alias_step_shapes(program: VcProgram, spec: StepSpec) -> None:
    if len(spec.inputs) != 1 or len(spec.outputs) != 1:
        raise ValueError(f"[plan] ALIAS expects 1 input and 1 output, got {spec.inputs} -> {spec.outputs}")
    inp = _tensor_shape_as_nchw(program.tensors[spec.inputs[0]])
    out = _tensor_shape_as_nchw(program.tensors[spec.outputs[0]])
    if _activation_size_bytes(inp) != _activation_size_bytes(out):
        raise ValueError(f"[plan] ALIAS byte size mismatch: in={inp} out={out}")


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _align_up(v: int, alignment: int) -> int:
    if alignment <= 0:
        return v
    return (v + alignment - 1) & ~(alignment - 1)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b
