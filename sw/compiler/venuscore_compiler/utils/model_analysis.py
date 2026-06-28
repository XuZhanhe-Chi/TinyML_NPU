# -*- coding: utf-8 -*-
"""
Module overview:
  - Rough model compute/size estimation utilities.
  - Dependencies:
    * Depends on: venuscore_compiler.ir.program, venuscore_compiler.ir.ops
    * Used by: analysis scripts and debugging tools
"""

from __future__ import annotations

from venuscore_compiler.ir.ops import VcConv2D, VcOp
from venuscore_compiler.ir.program import VcProgram


def estimate_macs(program: VcProgram) -> int:
    """Estimate MAC count for the program (rough heuristic)."""

    macs = 0
    for op in program.ops:
        if isinstance(op, VcConv2D):
            macs += op.kernel[0] * op.kernel[1]
    return macs


def estimate_model_size(program: VcProgram) -> int:
    """Estimate total tensor size (elements)."""

    return sum(t.num_elements() for t in program.tensors.values())
