"""
Tests for VenusCore uOP encoder/decoder round-trip semantics.
"""

import pytest

from venuscore_compiler.isa.decoder import decode_uop
from venuscore_compiler.isa.encoder import encode_uop
from venuscore_compiler.isa.layout_spec import Activation, Opcode, QMode
from venuscore_compiler.isa.uop_format import UOP_SIZE_BYTES, Uop


def test_encode_decode_roundtrip_default():
    """Default NOP should round-trip with default activation/qmode."""

    uop = Uop(opcode=Opcode.NOP)
    blob = encode_uop(uop)
    assert len(blob) == UOP_SIZE_BYTES
    decoded = decode_uop(blob)
    assert isinstance(decoded, Uop)
    assert decoded.opcode == Opcode.NOP
    assert decoded.act == Activation.NONE
    assert decoded.qmode == QMode.INT8


def test_encode_decode_with_fields():
    """Semantic fields should survive encode->decode."""

    uop = Uop(
        opcode=Opcode.CONV2D,
        act=Activation.RELU,
        qmode=QMode.INT8,
        first_flag=True,
        last_flag=True,
        sync=True,
        h_tile=5,
        w_tile=32,
        c4_in=2,
        c4_out=3,
        y_index=4,
        stride_h=2,
        stride_w=2,
        pad_top=1,
        pad_bottom=0,
        pad_left=1,
        pad_right=0,
        fi_stride=128,
        fo_stride=256,
        param_addr=0x1000,
        fi_addr=0x2000,
        fo_addr=0x3000,
    )
    blob = encode_uop(uop)
    decoded = decode_uop(blob)

    assert decoded.opcode == Opcode.CONV2D
    assert decoded.act == Activation.RELU
    assert decoded.qmode == QMode.INT8
    assert decoded.first_flag is True
    assert decoded.last_flag is True
    assert decoded.sync is True
    assert decoded.h_tile == 5
    assert decoded.w_tile == 32
    assert decoded.c4_in == 2
    assert decoded.c4_out == 3
    assert decoded.y_index == 4
    assert decoded.stride_h == 2 and decoded.stride_w == 2
    assert decoded.pad_top == 1
    assert decoded.pad_left == 1
    assert decoded.fi_stride == 128
    assert decoded.fo_stride == 256
    assert decoded.param_addr == 0x1000
    assert decoded.fi_addr == 0x2000
    assert decoded.fo_addr == 0x3000


def test_invalid_stride_raises():
    """Unsupported stride codes should raise during encoding."""

    uop = Uop(opcode=Opcode.CONV2D, stride_h=3, stride_w=3)
    with pytest.raises(ValueError):
        encode_uop(uop)
