# -*- coding: utf-8 -*-
"""
Bitfield layout and enums for VenusCore 32-byte uOPs (H+Y version).

This module is the single source of truth for:
  - Opcode / Activation / QMode enums (aligned with sw/compiler/doc/VenusCore_ISA.md).
  - UOP_FIELDS: mapping from semantic field names to (word, lsb, width).
  - pack_fields / unpack_fields helpers to translate between semantic fields
    and the eight 32-bit words (W0–W7) that make up a uOP.

Word meanings per ISA:
  W0: OPCODE, ACT_TYPE, FIRST_FLAG, LAST_FLAG, STRIDE, PAD_*, H_TILE, W_TILE, SYNC
  W1: C4_IN, C4_OUT, Y_INDEX, QMODE
  W2: FI_STRIDE, FO_STRIDE
  W3: PARAM_ADDR
  W4: FI_ADDR
  W5: FO_ADDR
  W6: IFM_W, IFM_H
  W7: ACTDMA_LINE_WORDS, OUTDMA_LINE_WORDS
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, Tuple

# Word indices for the 8-word uOP encoding.
WORD0 = 0
WORD1 = 1
WORD2 = 2
WORD3 = 3
WORD4 = 4
WORD5 = 5
WORD6 = 6
WORD7 = 7


class Opcode(IntEnum):
    """Opcode values defined by sw/compiler/doc/VenusCore_ISA.md."""

    NOP = 0x0
    CONV2D = 0x1
    PWCONV = 0x2
    DWCONV = 0x3
    AVGPOOL = 0x4
    MAXPOOL = 0x5
    MATMUL_FC = 0x6
    # 0x7–0xF are reserved; leave undefined to surface errors for invalid codes.


class Activation(IntEnum):
    """Activation type encoding (ACT_TYPE) as per ISA."""

    NONE = 0
    RELU = 1
    RELU6 = 2
    # 3–7 are reserved.


class QMode(IntEnum):
    """Weight quantization mode (QMODE) encoding."""

    INT8 = 0
    INT4 = 1
    INT2 = 2
    RESERVED = 3


# Mapping from semantic field name to (word_index, lsb_offset, bit_width).
UOP_FIELDS: Dict[str, Tuple[int, int, int]] = {
    # W0
    "opcode": (WORD0, 0, 4),
    "act_type": (WORD0, 4, 3),
    "first_flag": (WORD0, 7, 1),
    "last_flag": (WORD0, 8, 1),
    "stride": (WORD0, 9, 2),
    "pad_top": (WORD0, 11, 1),
    "pad_bottom": (WORD0, 12, 1),
    "pad_left": (WORD0, 13, 1),
    "pad_right": (WORD0, 14, 1),
    "h_tile": (WORD0, 15, 8),
    "w_tile": (WORD0, 23, 8),
    "sync": (WORD0, 31, 1),
    # W1
    "c4_in": (WORD1, 0, 10),
    "c4_out": (WORD1, 10, 10),
    "y_index": (WORD1, 20, 10),
    "qmode": (WORD1, 30, 2),
    # W2
    "fi_stride": (WORD2, 0, 16),
    "fo_stride": (WORD2, 16, 16),
    # W3–W5 full words
    "param_addr": (WORD3, 0, 32),
    "fi_addr": (WORD4, 0, 32),
    "fo_addr": (WORD5, 0, 32),
    # W6: IFM_W/IFM_H
    "ifm_w": (WORD6, 0, 16),
    "ifm_h": (WORD6, 16, 16),
    # W7: DMA precalc hints (word counts)
    "actdma_line_words": (WORD7, 0, 16),
    "outdma_line_words": (WORD7, 16, 16),
}


def _check_range(name: str, value: int, min_v: int, max_v: int) -> None:
    if not (min_v <= value <= max_v):
        raise ValueError(f"Field '{name}' value {value} out of range [{min_v}, {max_v}]")


def pack_fields(fields: Dict[str, int]) -> Dict[str, int]:
    """
    Pack semantic field values into eight 32-bit words according to UOP_FIELDS.

    Args:
        fields: dict containing all semantic fields:
            opcode, act_type, first_flag, last_flag, stride, pad_top, pad_bottom,
            pad_left, pad_right, h_tile, w_tile, sync, c4_in, c4_out, y_index,
            qmode, fi_stride, fo_stride, param_addr, fi_addr, fo_addr.

    Returns:
        Dict[str, int]: {"W0": int, ..., "W7": int} ready for struct packing.

    Raises:
        ValueError: if any field is out of the ISA-defined range.
        KeyError: if a required field is missing.
    """

    words = {
        "W0": 0,
        "W1": 0,
        "W2": 0,
        "W3": 0,
        "W4": 0,
        "W5": 0,
        "W6": 0,
        "W7": 0,
    }

    # Range validation per ISA.
    _check_range("opcode", fields["opcode"], 0, 0xF)
    _check_range("act_type", fields["act_type"], 0, 0x7)
    _check_range("first_flag", fields["first_flag"], 0, 1)
    _check_range("last_flag", fields["last_flag"], 0, 1)
    _check_range("stride", fields["stride"], 0, 3)
    _check_range("pad_top", fields["pad_top"], 0, 1)
    _check_range("pad_bottom", fields["pad_bottom"], 0, 1)
    _check_range("pad_left", fields["pad_left"], 0, 1)
    _check_range("pad_right", fields["pad_right"], 0, 1)
    _check_range("h_tile", fields["h_tile"], 0, 0xFF)
    _check_range("w_tile", fields["w_tile"], 0, 0xFF)
    _check_range("sync", fields["sync"], 0, 1)
    _check_range("c4_in", fields["c4_in"], 0, 0x3FF)
    _check_range("c4_out", fields["c4_out"], 0, 0x3FF)
    _check_range("y_index", fields["y_index"], 0, 0x3FF)
    _check_range("qmode", fields["qmode"], 0, 0x3)
    _check_range("fi_stride", fields["fi_stride"], 0, 0xFFFF)
    _check_range("fo_stride", fields["fo_stride"], 0, 0xFFFF)
    _check_range("param_addr", fields["param_addr"], 0, 0xFFFFFFFF)
    _check_range("fi_addr", fields["fi_addr"], 0, 0xFFFFFFFF)
    _check_range("fo_addr", fields["fo_addr"], 0, 0xFFFFFFFF)
    _check_range("ifm_w", fields["ifm_w"], 0, 0xFFFF)
    _check_range("ifm_h", fields["ifm_h"], 0, 0xFFFF)
    _check_range("actdma_line_words", fields["actdma_line_words"], 0, 0xFFFF)
    _check_range("outdma_line_words", fields["outdma_line_words"], 0, 0xFFFF)

    def _set_bits(word_value: int, offset: int, width: int, value: int) -> int:
        mask = ((1 << width) - 1) << offset
        return (word_value & ~mask) | ((value << offset) & mask)

    # Deterministic order ensures consistent packing.
    for name in [
        "opcode",
        "act_type",
        "first_flag",
        "last_flag",
        "stride",
        "pad_top",
        "pad_bottom",
        "pad_left",
        "pad_right",
        "h_tile",
        "w_tile",
        "sync",
        "c4_in",
        "c4_out",
        "y_index",
        "qmode",
        "fi_stride",
        "fo_stride",
        "param_addr",
        "fi_addr",
        "fo_addr",
        "ifm_w",
        "ifm_h",
        "actdma_line_words",
        "outdma_line_words",
    ]:
        word_idx, offset, width = UOP_FIELDS[name]
        word_key = f"W{word_idx}"
        words[word_key] = _set_bits(words[word_key], offset, width, fields[name])

    return words


def unpack_fields(words: Dict[str, int]) -> Dict[str, int]:
    """
    Unpack eight 32-bit words into semantic field values using UOP_FIELDS.

    Args:
        words: dict {"W0": int, ..., "W7": int}.

    Returns:
        Dict[str, int]: semantic field values keyed by field name.
    """

    def _get_bits(word_value: int, offset: int, width: int) -> int:
        mask = (1 << width) - 1
        return (word_value >> offset) & mask

    fields: Dict[str, int] = {}
    for name, (word_idx, offset, width) in UOP_FIELDS.items():
        word_key = f"W{word_idx}"
        fields[name] = _get_bits(words[word_key], offset, width)
    return fields
