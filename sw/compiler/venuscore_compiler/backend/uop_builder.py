# -*- coding: utf-8 -*-
"""
Wrapper for building semantic uOPs (aligned to design doc naming).

The main entry point is :func:`build_uops`, which converts a TilePlan and
MemoryPlan into a list of semantic :class:`Uop` objects. The actual mapping
logic lives in :mod:`backend.codegen_uop`.
"""

from __future__ import annotations

from typing import List

from venuscore_compiler.backend.codegen_uop import generate_uops
from venuscore_compiler.backend.memory_planner import MemoryPlan
from venuscore_compiler.isa.uop_format import Uop
from venuscore_compiler.midend.types import TilePlan


__all__ = ["build_uops"]


def build_uops(
    tile_plan: TilePlan,
    memory_plan: MemoryPlan,
    target: str = "venuscore-v1",
) -> List[Uop]:
    """
    Build semantic uOPs for a given TilePlan and MemoryPlan.

    This is a thin alias to :func:`backend.codegen_uop.generate_uops`.

    Args:
        tile_plan:
            Midend tiling result.
        memory_plan:
            Backend memory planning result.
        target:
            Reserved target name for future extensions; currently unused.

    Returns:
        A list of semantic :class:`Uop` objects ready to be encoded.
    """
    return generate_uops(tile_plan, memory_plan, target=target)
