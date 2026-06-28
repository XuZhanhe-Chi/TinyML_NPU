# -*- coding: utf-8 -*-
"""
Module overview:
  - Top-level VenusCore compiler entrypoint, wiring frontend/midend/backend and artifact emission.
  - Dependencies:
    * Depends on: backend.*, utils.debug_dump
    * Used by: cli.py and examples/* via compile_program
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .version import __version__

if TYPE_CHECKING:  # pragma: no cover
    from venuscore_compiler.ir.program import VcProgram
    from venuscore_compiler.runtime.binary_format import CompiledArtifact

__all__ = ["compile_program", "__version__"]


def compile_program(
    program: "VcProgram",
    output_dir: str | Path,
    target: str = "venuscore-v1",
    dump_ir: bool = False,
    dump_uop: bool = False,
    ifm_base: int | None = 0x0000_0000,
    ofm_base: int | None = None,
    param_base: int | None = 0x0010_0000,
    hw: "HwConfig | None" = None,
    emit_plan: bool = True,
    emit_alias_steps: bool = False,
    post_check: bool = True,
    post_check_act_base: int = 0x2000_0000,
    post_check_param_base: int = 0x2100_0000,
) -> "CompiledArtifact":
    """
    Run a minimal VenusCore compilation pipeline.

    Optional address parameters let callers constrain the usable address ranges for IFM/OFM/Param blobs.
    """

    from venuscore_compiler.backend import codegen_uop, layout_param_block, memory_planner
    from venuscore_compiler.config import default_hw_config, HwConfig
    from venuscore_compiler.midend import (
        check_layer_constraints,
        check_tile_constraints,
        compute_quant_params,
        tile_program,
        normalize_program,
        build_param_data,
        prepare_param_buffers,
        layout_lowering,
    )
    from venuscore_compiler.midend.partition_fallback import partition_program
    from venuscore_compiler.runtime.binary_format import CompiledArtifact
    from venuscore_compiler.utils import debug_dump

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if hw is None:
        hw = default_hw_config()

    # Addressing mode:
    # - If all bases are None, emit uOP addresses as offsets from 0.
    #   A host/SoC can relocate by adding a chosen base address.
    # - Otherwise, use the provided absolute byte addresses.
    address_mode = "absolute"
    if ifm_base is None and ofm_base is None and param_base is None:
        address_mode = "offset"
        ifm_base = 0
        ofm_base = None
        # In offset mode, PARAM_ADDR values are offsets within the param blob.
        # A host/SoC can relocate by adding its chosen PARAM region base.
        param_base = 0
    else:
        # Keep legacy behavior: ofm_base may be None to enable ping-pong placement.
        if ifm_base is None or param_base is None:
            raise ValueError(
                "ifm_base and param_base must be provided unless using offset mode "
                "(set ifm_base=ofm_base=param_base=None)."
            )

    # Validate IR against basic hardware limits before spending effort on later stages.
    normalize_program(program)
    check_layer_constraints(program, target=target)

    step_specs = None
    if emit_plan:
        step_specs = partition_program(program)

    def _needs_arena_allocation() -> bool:
        # Any CPU step implies multi-input / fan-out that ping-pong cannot safely represent.
        if step_specs is not None and any(s.step_type.name == "CPU" for s in step_specs):
            return True
        # Conservative structural check: fan-out (a tensor consumed by >1 op) or multi-input ops.
        consumers: dict[str, int] = {}
        for op in program.ops:
            if len(getattr(op, "inputs", []) or []) > 1:
                return True
            for t in getattr(op, "inputs", []) or []:
                consumers[str(t)] = consumers.get(str(t), 0) + 1
        return any(v > 1 for v in consumers.values())

    # Layout lowering to produce per-op geometry info.
    layout_info = layout_lowering.lower_layouts(program)

    # Produce a tile plan; currently one tile per op but hook point for real tiling heuristics.
    tile_plan = tile_program(program, layout_info=layout_info, hw=hw, target=target)
    check_tile_constraints(tile_plan, program, target=target, hw=hw)

    quant_table, qmode_map = compute_quant_params(program)

    # Attach quant_table/qmode into layer configs.
    for layer_cfg in tile_plan.layers:
        if layer_cfg.name in quant_table:
            layer_cfg.quant_table = quant_table[layer_cfg.name]
        if layer_cfg.name in qmode_map:
            layer_cfg.qmode = qmode_map[layer_cfg.name]

    # Assign linear offsets for tensors so later stages can reference concrete addresses.
    memory_plan = memory_planner.plan_memory(
        tile_plan,
        ifm_base=ifm_base,
        ofm_base=ofm_base,
        param_base=param_base,
        allocation_mode="arena" if (emit_plan and _needs_arena_allocation()) else "ping_pong",
        program=program if emit_plan else None,
    )

    # Lower tiles into logical uOPs, then encode them into binary blobs for the runtime.
    uops = codegen_uop.generate_uops(tile_plan, memory_plan, target=target)
    uop_binary = codegen_uop.encode_uops(uops)

    # Build parameter blocks (weights/bias/quant) using the memory layout decisions.
    param_data = prepare_param_buffers(build_param_data(program), tile_plan)
    param_block = layout_param_block.build_param_block(tile_plan, memory_plan, param_data, quant_table, target=target)

    artifact = CompiledArtifact(
        uops=uops,
        uop_binary=uop_binary,
        param_block=param_block,
        metadata={
            "target": target,
            "address_mode": address_mode,
            "tile_uop_map": _build_tile_uop_map(tile_plan, uops),
            "layer_quant": _build_layer_quant_snapshot(tile_plan),
            "activation_peak_bytes": memory_plan.activation_peak_bytes,
            "weight_bytes": len(param_block),
            "output_base": _get_output_base(tile_plan, memory_plan),
            "input_base": _get_input_base(tile_plan, memory_plan, int(ifm_base)),
            "input_size": _get_input_size(tile_plan),
            "output_size": _get_output_size(tile_plan),
            "param_base": memory_plan.param_base,
        },
    )

    if emit_plan:
        from venuscore_compiler.plan.builder import build_plan

        plan = build_plan(
            program=program,
            tile_plan=tile_plan,
            memory_plan=memory_plan,
            uops_len_bytes=len(uop_binary),
            params_len_bytes=len(param_block),
            step_specs=step_specs,
            emit_alias_steps=emit_alias_steps,
        )
        artifact.metadata["plan"] = plan.to_dict()
        artifact.metadata["plan_step_count"] = len(plan.steps)
        artifact.metadata["plan_tensor_count"] = len(plan.tensors)
        artifact.metadata["plan_npu_step_count"] = sum(1 for s in plan.steps if getattr(s, "step_type", None).name == "NPU")
        artifact.metadata["plan_cpu_step_count"] = sum(1 for s in plan.steps if getattr(s, "step_type", None).name == "CPU")
        artifact.metadata["plan_alias_step_count"] = sum(1 for s in plan.steps if getattr(s, "step_type", None).name == "ALIAS")
        artifact.metadata["plan_arena_bytes"] = int(plan.arena_bytes)

    # Optional human-readable debug dumps to help trace IR and uOP emission.
    if dump_ir:
        debug_dump.dump_program_to_json(program, output_path / "debug_ir.json")
    if dump_uop:
        debug_dump.dump_uops(uops, output_path / "debug_uops.json")

    artifact.save_to_directory(output_path)

    if post_check:
        from venuscore_compiler.runtime.post_checks import run_post_compile_checks

        run_post_compile_checks(
            output_path,
            act_base=post_check_act_base,
            param_base=post_check_param_base,
        )
    return artifact


def _build_tile_uop_map(tile_plan, uops) -> list[dict[str, object]]:
    """
    Build a simple mapping from tile to uOP index for debugging/metadata.

    Assumes generate_uops followed tile_plan iteration order (op order, then tile order).
    """

    mapping: list[dict[str, object]] = []
    idx = 0
    for op_name, tiles in tile_plan.tiles_by_op.items():
        for tile in tiles:
            mapping.append({"op": op_name, "tile_id": tile.tile_id, "uop_index": idx})
            idx += 1
    return mapping


def _build_layer_quant_snapshot(tile_plan) -> list[dict[str, object]]:
    """
    Extract per-layer quantization parameters for debugging/metadata.

    The snapshot includes qmode and the full quant_table (list of [bias, scale, shift]).
    """

    layers = []
    for cfg in getattr(tile_plan, "layers", []):
        layers.append(
            {
                "name": getattr(cfg, "name", ""),
                "op_type": getattr(cfg, "op_type", ""),
                "qmode": getattr(cfg, "qmode", None),
                "quant_table": getattr(cfg, "quant_table", []),
            }
        )
    return layers


def _get_output_base(tile_plan, memory_plan) -> int | None:
    """Return the base address of the final OFM tensor if known."""

    if not getattr(tile_plan, "layers", None):
        return None
    last_layer = tile_plan.layers[-1]
    ofm_name = getattr(last_layer, "ofm_name", None)
    if ofm_name and ofm_name in memory_plan.ofm_offsets:
        return memory_plan.ofm_offsets[ofm_name]
    return None


def _get_input_base(tile_plan, memory_plan, default_base: int) -> int:
    """Return the base address of the first IFM tensor, falling back to provided base."""

    if getattr(tile_plan, "layers", None):
        first = tile_plan.layers[0]
        if first.ifm_name and first.ifm_name in memory_plan.ifm_offsets:
            return memory_plan.ifm_offsets[first.ifm_name]
    return default_base


def _get_input_size(tile_plan) -> int:
    """Estimate input feature map size in bytes for the first layer."""

    if not getattr(tile_plan, "layers", None):
        return 0
    first = tile_plan.layers[0]
    c4 = getattr(first, "c4_in", 0)
    h = getattr(first, "ifm_h", 0)
    w = getattr(first, "ifm_w", 0)
    return c4 * h * w * 4  # NCHWc4 int8 => WORD_BYTES=4


def _get_output_size(tile_plan) -> int:
    """Estimate output feature map size in bytes for the final layer."""

    if not getattr(tile_plan, "layers", None):
        return 0
    last = tile_plan.layers[-1]
    c4 = getattr(last, "c4_out", 0)
    h = getattr(last, "ofm_h", 0)
    w = getattr(last, "ofm_w", 0)
    return c4 * h * w * 4  # NCHWc4 int8 => WORD_BYTES=4
