# -*- coding: utf-8 -*-
"""
Runtime execution plan (Plan) for mixed NPU/CPU graphs.

This package defines a compact plan representation that can be embedded into
`bundle.h` and consumed by firmware:

  - TensorDesc: runtime tensor placement in an activation arena (offset/shape).
  - StepDesc:   ordered execution steps (NPU / CPU / ALIAS).

The plan is intentionally minimal:
  - The physical activation layout is assumed to be NCHWc4 int8 for v1.
  - NPU steps reference sub-ranges of the global uops/params blobs (word units).
  - CPU steps reference small built-in kernels (e.g., ADD / CONCAT_C).
  - ALIAS steps represent zero-copy view changes (identity/reshape) and do not
    allocate new memory.
"""

from .types import (
    AliasStepDesc,
    CpuKernel,
    CpuStepDesc,
    NpuStepDesc,
    Plan,
    StepDesc,
    StepType,
    TensorDesc,
)

__all__ = [
    "TensorDesc",
    "StepType",
    "StepDesc",
    "NpuStepDesc",
    "CpuKernel",
    "CpuStepDesc",
    "AliasStepDesc",
    "Plan",
]

