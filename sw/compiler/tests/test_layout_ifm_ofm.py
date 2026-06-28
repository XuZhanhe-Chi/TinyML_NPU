# -*- coding: utf-8 -*-
"""
Unit tests for NCHWc4 IFM/OFM offset helpers.
"""

from __future__ import annotations

import pytest

from venuscore_compiler.backend.layout_ifm_ofm import compute_ifm_offset, compute_ofm_offset
from venuscore_compiler.ir.tensor import VcTensor


def test_ifm_ofm_offsets_nchwc4() -> None:
    """Check offsets for a small 1x1x2x4 tensor in NCHW layout."""

    tensor = VcTensor(name="ifm", shape=(1, 2, 2, 4), layout="NCHW", dtype="int8")
    # Channel 0, h=0, w=0 should be byte 0
    assert compute_ifm_offset(tensor, (0, 0, 0, 0)) == 0
    # Channel 1 (same spatial), packed into same word at byte offset 1
    assert compute_ifm_offset(tensor, (0, 1, 0, 0)) == 1
    # Next spatial element (h=0, w=1) advances word_index by 1 word (4 bytes)
    assert compute_ifm_offset(tensor, (0, 0, 0, 1)) == 4
    # compute_ofm_offset mirrors IFM helper
    assert compute_ofm_offset(tensor, (0, 1, 0, 1)) == compute_ifm_offset(tensor, (0, 1, 0, 1))


def test_ifm_offset_bounds_check() -> None:
    """Ensure assertions fire on out-of-bounds or wrong dtype."""

    tensor = VcTensor(name="ifm", shape=(1, 1, 1, 1), layout="NCHW", dtype="int8")
    with pytest.raises(AssertionError):
        compute_ifm_offset(tensor, (0, 1, 0, 0))
    tensor_bad = VcTensor(name="ifm", shape=(1, 1, 1, 1), layout="NCHW", dtype="float32")
    with pytest.raises(AssertionError):
        compute_ifm_offset(tensor_bad, (0, 0, 0, 0))
