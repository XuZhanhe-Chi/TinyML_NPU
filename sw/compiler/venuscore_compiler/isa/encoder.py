# -*- coding: utf-8 -*-
"""
Encoder for VenusCore uOPs (semantic -> encoded 32-byte form).
"""

from __future__ import annotations

import struct
from typing import Iterable, List

from venuscore_compiler.isa.layout_spec import Activation, QMode, pack_fields
from venuscore_compiler.isa.uop_format import UOP_SIZE_BYTES, Uop


def encode_uop(uop: Uop) -> bytes:
    """
    Encode a single uOP into a 32-byte little-endian blob using layout_spec bitfields.

    Stride encoding follows ISA: 1 -> stride 1, 2 -> stride 2; other codes are illegal.
    """

    if uop.stride_h not in (1, 2):
        raise ValueError(f"Unsupported stride_h {uop.stride_h}; ISA only supports 1 or 2.")
    if uop.stride_w != uop.stride_h:
        raise ValueError("VenusCore ISA currently assumes stride_h == stride_w.")
    stride_code = 1 if uop.stride_h == 1 else 2

    fields = {
        "opcode": int(uop.opcode.value),
        "act_type": int(uop.act.value if uop.act is not None else Activation.NONE.value),
        "first_flag": 1 if uop.first_flag else 0,
        "last_flag": 1 if uop.last_flag else 0,
        "stride": stride_code,
        "pad_top": uop.pad_top,
        "pad_bottom": uop.pad_bottom,
        "pad_left": uop.pad_left,
        "pad_right": uop.pad_right,
        "h_tile": uop.h_tile,
        "w_tile": uop.w_tile,
        "sync": 1 if uop.sync else 0,
        "c4_in": uop.c4_in,
        "c4_out": uop.c4_out,
        "y_index": uop.y_index,
        "qmode": int(uop.qmode.value if uop.qmode is not None else QMode.INT8.value),
        "fi_stride": uop.fi_stride,
        "fo_stride": uop.fo_stride,
        "param_addr": uop.param_addr,
        "fi_addr": uop.fi_addr,
        "fo_addr": uop.fo_addr,
        "ifm_w": uop.ifm_w,
        "ifm_h": uop.ifm_h,
        "actdma_line_words": uop.actdma_line_words,
        "outdma_line_words": uop.outdma_line_words,
    }

    packed_words = pack_fields(fields)
    words_tuple = (
        packed_words["W0"],
        packed_words["W1"],
        packed_words["W2"],
        packed_words["W3"],
        packed_words["W4"],
        packed_words["W5"],
        packed_words["W6"],
        packed_words["W7"],
    )
    return struct.pack("<IIIIIIII", *words_tuple)


def encode_uops(uops: Iterable[Uop]) -> bytes:
    """Encode a sequence of uOPs into a contiguous bytes object."""

    blobs: List[bytes] = [encode_uop(uop) for uop in uops]
    return b"".join(blobs)
