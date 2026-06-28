# -*- coding: utf-8 -*-
"""
Midend pipeline: layer-level checks, tiling, and tile-level checks.

This stage sits between the frontend IR construction and the backend codegen:
  - check_layer_constraints: validate hardware capability at layer/IR level.
  - tile_program: produce TilePlan (logical uOP units without addresses).
  - check_tile_constraints: validate per-tile capacity/geometry constraints.
"""

from __future__ import annotations

from venuscore_compiler.midend.npu_constraints import check_layer_constraints, check_tile_constraints
from venuscore_compiler.midend.quantize import QuantTable, QModeMap, compute_quant_params
from venuscore_compiler.midend.tiler import tile_program
from venuscore_compiler.midend.types import LayerConfig, TileDesc, TilePlan
from venuscore_compiler.midend.normalize import normalize_program
from venuscore_compiler.midend.param_data import build_param_data, prepare_param_buffers, ParamData
from venuscore_compiler.midend import layout_lowering
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.config import HwConfig, default_hw_config

__all__ = [
    "check_layer_constraints",
    "tile_program",
    "check_tile_constraints",
    "TilePlan",
    "LayerConfig",
    "TileDesc",
    "compute_quant_params",
    "QuantTable",
    "QModeMap",
    "normalize_program",
    "build_param_data",
    "prepare_param_buffers",
    "ParamData",
    "run_midend",
]


def run_midend(
    program: VcProgram,
    target: str = "venuscore-v1",
    hw: HwConfig | None = None,
) -> tuple[TilePlan, QuantTable, QModeMap, ParamData]:
    """
    End-to-end midend pipeline producing TilePlan and quantization metadata.

    Steps:
      1) normalize_program
      2) check_layer_constraints
      3) lower_layouts
      4) tile_program
      5) check_tile_constraints
      6) compute_quant_params (and attach to LayerConfig)
      7) build_param_data / prepare_param_buffers
    """
    if hw is None:
        hw = default_hw_config()

    normalize_program(program)
    check_layer_constraints(program, target=target)

    layout_info = layout_lowering.lower_layouts(program)
    tile_plan = tile_program(program, layout_info=layout_info, hw=hw, target=target)
    check_tile_constraints(tile_plan, program, target=target, hw=hw)

    quant_table, qmode_map = compute_quant_params(program)
    # Attach quantization metadata into LayerConfig for backend convenience.
    for layer_cfg in tile_plan.layers:
        if layer_cfg.name in quant_table:
            layer_cfg.quant_table = quant_table[layer_cfg.name]
        if layer_cfg.name in qmode_map:
            layer_cfg.qmode = qmode_map[layer_cfg.name]

    param_data = prepare_param_buffers(build_param_data(program), tile_plan)
    return tile_plan, quant_table, qmode_map, param_data
