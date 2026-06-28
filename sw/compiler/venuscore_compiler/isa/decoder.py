# -*- coding: utf-8 -*-
"""
Decoder for VenusCore uOPs (32-byte -> semantic Uop).
"""

from __future__ import annotations

import struct
from typing import List

from venuscore_compiler.isa.layout_spec import Activation, Opcode, QMode, unpack_fields
from venuscore_compiler.isa.uop_format import UOP_SIZE_BYTES, Uop


def decode_uop(blob: bytes) -> Uop:
    """Decode a single 32-byte uOP into a semantic Uop."""

    if len(blob) != UOP_SIZE_BYTES:
        raise ValueError(f"Expected {UOP_SIZE_BYTES} bytes, got {len(blob)}")
    w0, w1, w2, w3, w4, w5, w6, w7 = struct.unpack("<IIIIIIII", blob)
    fields = unpack_fields(
        {"W0": w0, "W1": w1, "W2": w2, "W3": w3, "W4": w4, "W5": w5, "W6": w6, "W7": w7}
    )

    flags = {}

    # Opcode
    opcode_raw = fields["opcode"]
    try:
        opcode = Opcode(opcode_raw)
    except ValueError:
        opcode = Opcode.NOP
        flags["opcode_raw"] = opcode_raw

    # Activation
    act_raw = fields["act_type"]
    if act_raw in (Activation.NONE.value, Activation.RELU.value, Activation.RELU6.value):
        act = Activation(act_raw)
    else:
        act = None
        flags["act_type_raw"] = act_raw

    # QMode
    qmode_raw = fields["qmode"]
    if qmode_raw in (QMode.INT8.value, QMode.INT4.value, QMode.INT2.value):
        qmode = QMode(qmode_raw)
    else:
        qmode = None
        flags["qmode_raw"] = qmode_raw

    # Stride decoding (0/3 are reserved and treated as errors)
    stride_code = fields["stride"]
    if stride_code == 1:
        stride_h = stride_w = 1
    elif stride_code == 2:
        stride_h = stride_w = 2
    else:
        raise ValueError(f"Illegal STRIDE encoding {stride_code} in uOP.")

    return Uop(
        opcode=opcode,
        act=act,
        qmode=qmode,
        first_flag=bool(fields["first_flag"]),
        last_flag=bool(fields["last_flag"]),
        sync=bool(fields["sync"]),
        h_tile=fields["h_tile"],
        w_tile=fields["w_tile"],
        c4_in=fields["c4_in"],
        c4_out=fields["c4_out"],
        y_index=fields["y_index"],
        ifm_w=fields["ifm_w"],
        ifm_h=fields["ifm_h"],
        actdma_line_words=fields["actdma_line_words"],
        outdma_line_words=fields["outdma_line_words"],
        stride_h=stride_h,
        stride_w=stride_w,
        pad_top=fields["pad_top"],
        pad_bottom=fields["pad_bottom"],
        pad_left=fields["pad_left"],
        pad_right=fields["pad_right"],
        fi_stride=fields["fi_stride"],
        fo_stride=fields["fo_stride"],
        param_addr=fields["param_addr"],
        fi_addr=fields["fi_addr"],
        fo_addr=fields["fo_addr"],
        flags=flags,
    )


def decode_uops(data: bytes) -> List[Uop]:
    """Decode a bytes object into a list of semantic Uops."""

    if len(data) % UOP_SIZE_BYTES != 0:
        raise ValueError("uOP byte stream length must be a multiple of UOP_SIZE_BYTES.")
    uops: List[Uop] = []
    for i in range(0, len(data), UOP_SIZE_BYTES):
        uops.append(decode_uop(data[i : i + UOP_SIZE_BYTES]))
    return uops
