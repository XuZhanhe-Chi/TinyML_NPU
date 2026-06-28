# -*- coding: utf-8 -*-
"""
Module overview:
  - Math helpers for convolution dimensions, padding, and related utilities.
  - Dependencies:
    * Depends on: stdlib only
    * Used by: backend.memory_planner, backend.tiler, and other layout modules
"""

from __future__ import annotations

from typing import Tuple


def compute_output_dim(input_size: int, kernel: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    """Compute the output dimension for 1D convolution/pooling."""

    # Standard conv formula: floor((N + 2P - dilation*(K-1) -1)/stride) + 1
    return ((input_size + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1


def pad_to_multiple(value: int, multiple: int) -> int:
    """Pad value up to the nearest multiple."""

    if multiple == 0:
        return value
    remainder = value % multiple
    return value if remainder == 0 else value + (multiple - remainder)


def compute_padding(input_size: int, output_size: int, kernel: int, stride: int) -> Tuple[int, int]:
    """Compute symmetric padding given output size."""

    total = max((output_size - 1) * stride + kernel - input_size, 0)
    return total // 2, total - total // 2
