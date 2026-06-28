# -*- coding: utf-8 -*-
"""
IFM/OFM element offset helpers for VenusCore (NCHWc4 layout, int8 activations).

This module provides helpers to compute **byte offsets** for IFM/OFM tensors
under the NCHWc4 packing scheme used by VenusCore:

    - Activations are int8 / uint8.
    - Channels are grouped by 4 into 32-bit words (WORD_BYTES = 4).
    - Logical tensor shape is (N, C, H, W).
    - Physical layout is:

        word_index =
            ((n * C4_total + c4) * FI_STRIDE) + (h * W_total) + w

        where:
            C4_total = ceil_div(C_total, 4)
            c4       = c // 4
            k        = c % 4   # intra-word channel index in [0, 3]
            FI_STRIDE = H_total * W_total (words per channel plane)

        byte_offset = word_index * WORD_BYTES + k

The functions here return offsets **relative to the tensor base address**.
The backend MemoryPlan is responsible for assigning the base addresses.
"""

from __future__ import annotations

from typing import Tuple

from venuscore_compiler.ir.tensor import VcTensor

# 1 word = 4 * int8
WORD_BYTES = 4


def _check_tensor_and_index(tensor: VcTensor, index: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Basic sanity checks on tensor shape / dtype and index ranges.

    Args:
        tensor: Logical tensor with shape (N, C, H, W).
        index:  (n, c, h, w) logical coordinates.

    Returns:
        The validated (n, c, h, w) tuple (same as input if all checks pass).

    Raises:
        AssertionError if any constraint is violated.
    """
    n, c, h, w = index

    assert tensor.dtype in ("int8", "uint8"), (
        "NCHWc4 offset helper expects int8/uint8 tensor, "
        f"got dtype={tensor.dtype!r}"
    )

    n_total, c_total, h_total, w_total = tensor.shape
    assert 0 <= n < n_total, f"n index out of range: {n} (N={n_total})"
    assert 0 <= c < c_total, f"c index out of range: {c} (C={c_total})"
    assert 0 <= h < h_total, f"h index out of range: {h} (H={h_total})"
    assert 0 <= w < w_total, f"w index out of range: {w} (W={w_total})"

    return n, c, h, w


def compute_ifm_offset(tensor: VcTensor, index: Tuple[int, int, int, int]) -> int:
    """
    Compute the byte offset relative to the IFM tensor base for an element.

    Layout assumptions:

    - Tensor is logical NCHW with int8/uint8 elements.
    - Physical layout is NCHWc4 (channels grouped by 4 into 32-bit words).
    - FI_STRIDE used here is the **logical** IFM_H * IFM_W (words per
      channel plane). The actual FI_STRIDE written into uOP is provided
      by the backend (LayerConfig) and should match this value.

    Args:
        tensor:
            Logical IFM tensor with shape (N, C, H, W), dtype int8/uint8.
        index:
            Logical coordinates (n, c, h, w).

    Returns:
        Byte offset relative to the tensor base address.
    """
    n, c, h, w = _check_tensor_and_index(tensor, index)

    n_total, c_total, h_total, w_total = tensor.shape

    # C4_total = ceil_div(C, 4)
    c4_total = (c_total + 3) >> 2
    fi_stride = h_total * w_total  # words per channel plane

    # Channel group index and intra-word channel index.
    c4 = c >> 2     # c // 4
    k = c & 3       # c % 4

    # Word index in the flattened NCHWc4 space.
    word_index = ((n * c4_total + c4) * fi_stride) + h * w_total + w

    # Byte offset = word_index * 4 + k (k is intra-word channel index in [0, 3]).
    return (word_index * WORD_BYTES) + k


def compute_ofm_offset(tensor: VcTensor, index: Tuple[int, int, int, int]) -> int:
    """
    Compute the byte offset relative to the OFM tensor base for an element.

    OFM uses the same NCHWc4 layout as IFM, so this is simply a thin wrapper
    around :func:`compute_ifm_offset`.

    Args:
        tensor:
            Logical OFM tensor with shape (N, C, H, W), dtype int8/uint8.
        index:
            Logical coordinates (n, c, h, w).

    Returns:
        Byte offset relative to the tensor base address.
    """
    return compute_ifm_offset(tensor, index)
