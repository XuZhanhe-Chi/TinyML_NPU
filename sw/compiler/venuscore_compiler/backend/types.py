# -*- coding: utf-8 -*-
"""
Backend type definitions (public façade).

This module provides a stable import location for the main backend-facing
types:

    - MemoryPlan:  backend.memory_planner.MemoryPlan
    - Uop:         isa.uop_format.Uop (semantic uOP)
    - UopSemantic: type alias to Uop (kept for compatibility with older code)

The intention is that other parts of the project can simply do:

    from venuscore_compiler.backend.types import MemoryPlan, Uop

without depending on the internal layout of backend.memory_planner or
venuscore_compiler.isa.uop_format.
"""

from __future__ import annotations

from venuscore_compiler.backend.memory_planner import MemoryPlan
from venuscore_compiler.isa.uop_format import Uop

# Backward-compatible alias: older code may still refer to UopSemantic.
UopSemantic = Uop

__all__ = ["MemoryPlan", "Uop", "UopSemantic"]
