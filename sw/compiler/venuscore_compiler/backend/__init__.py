# -*- coding: utf-8 -*-
"""
Public backend API for the VenusCore compiler.

This package exposes a small, stable set of high-level entry points:

- plan_memory(tile_plan, ... ) -> MemoryPlan
    Naive linear memory planner for IFM / OFM / Param Block regions.

- build_param_blocks(tile_plan, memory_plan, param_data, quant_table=None)
    Construct the contiguous Param Block blob (params.bin) according to
    the VenusCore data layout specification.

- build_uops(tile_plan, memory_plan, target="venuscore-v1") -> List[Uop]
    Generate semantic uOPs from TilePlan + MemoryPlan.

- encode_uops_bytes(uops) -> bytes
    Encode semantic uOPs into their 32B binary representation using the
    ISA encoder.

Typical usage:

    from venuscore_compiler.backend import (
        MemoryPlan,
        plan_memory,
        build_param_blocks,
        build_uops,
        encode_uops_bytes,
    )

    mp = plan_memory(tile_plan)
    params_bin = build_param_blocks(tile_plan, mp, param_data, quant_table)
    uops = build_uops(tile_plan, mp)
    uops_bin = encode_uops_bytes(uops)
"""

from __future__ import annotations

from .memory_planner import MemoryPlan, plan_memory
from .param_block import build_param_blocks
from .uop_builder import build_uops
from .uop_encoder import encode_uops_bytes

__all__ = [
    "MemoryPlan",
    "plan_memory",
    "build_param_blocks",
    "build_uops",
    "encode_uops_bytes",
]
