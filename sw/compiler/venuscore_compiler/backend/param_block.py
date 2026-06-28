# -*- coding: utf-8 -*-
"""
Thin wrapper for Param Block construction aligned to design doc naming.

This module exposes a stable API:

    build_param_blocks(tile_plan, memory_plan, param_data, quant_table, target)

and delegates the actual layout work to backend.layout_param_block, which
implements the Param Block format specified in the VenusCore data layout
document:

    [ Quant Coeff Block (Cout_tile * 8B) ]
    [ 16B alignment padding ]
    [ Weight Data Block (op_type-specific) ]
"""

from __future__ import annotations

from typing import Any, Dict

from venuscore_compiler.backend.layout_param_block import build_param_block as _build
from venuscore_compiler.backend.memory_planner import MemoryPlan
from venuscore_compiler.midend.types import TilePlan
from venuscore_compiler.midend.quantize import QuantTable


__all__ = ["build_param_blocks"]


def build_param_blocks(
    tile_plan: TilePlan,
    memory_plan: MemoryPlan,
    param_data: Dict[str, Dict[str, Any]],
    quant_table: QuantTable | None = None,
    target: str = "venuscore-v1",
) -> bytes:
    """
    Build the full Param Block blob (params.bin) for a TilePlan.

    This is a convenience wrapper around
    :func:`backend.layout_param_block.build_param_block`.

    Args:
        tile_plan:
            Tiling plan with tiles_by_op and LayerConfig list.
        memory_plan:
            MemoryPlan with param_base and per-tile param_offsets prepared.
        param_data:
            Mapping from tensor name to a dict with at least:
              - "shape": List[int]
              - "data":  flat List[int] of int8 / int32 etc.
        quant_table:
            Global quant table, if any. LayerConfig.quant_table takes priority.
        target:
            Reserved target name for future extensions; currently unused.

    Returns:
        params.bin content as bytes.
    """
    return _build(tile_plan, memory_plan, param_data, quant_table, target=target)
