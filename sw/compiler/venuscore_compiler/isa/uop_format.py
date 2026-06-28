# -*- coding: utf-8 -*-
"""
Semantic VenusCore uOP representation aligned with sw/compiler/doc/VenusCore_ISA.md (32-byte uOP, H+Y version).

Mapping to encoded words:
  W0: OPCODE, ACT_TYPE, FIRST_FLAG, LAST_FLAG, STRIDE, PAD_*, H_TILE, W_TILE, SYNC
  W1: C4_IN, C4_OUT, Y_INDEX, QMODE
  W2: FI_STRIDE, FO_STRIDE
  W3: PARAM_ADDR
  W4: FI_ADDR
  W5: FO_ADDR
  W6: IFM_W, IFM_H
  W7: DMA_PRECALC (ACTDMA_LINE_WORDS, OUTDMA_LINE_WORDS)

Encoder/decoder are responsible for packing/unpacking these semantic fields into the eight 32-bit words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from venuscore_compiler.isa.layout_spec import Activation, Opcode, QMode

UOP_SIZE_BYTES = 32


@dataclass
class Uop:
    """Semantic VenusCore micro-op; fields correspond directly to ISA semantics."""

    opcode: Opcode = Opcode.NOP
    act: Optional[Activation] = None
    qmode: Optional[QMode] = None

    # Layer / tile flags
    first_flag: bool = False
    last_flag: bool = False
    sync: bool = False

    # Tile geometry
    h_tile: int = 0
    w_tile: int = 0
    c4_in: int = 0
    c4_out: int = 0
    y_index: int = 0
    ifm_w: int = 0
    ifm_h: int = 0

    # DMA precalc hints (word counts, not bytes)
    actdma_line_words: int = 0
    outdma_line_words: int = 0

    # Stride and padding (stride_h == stride_w required by ISA)
    stride_h: int = 1
    stride_w: int = 1
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0

    # Strides and base addresses (logical values per ISA; encoder handles bit packing)
    fi_stride: int = 0
    fo_stride: int = 0
    param_addr: int = 0
    fi_addr: int = 0
    fo_addr: int = 0

    # Auxiliary metadata not encoded into hardware bits.
    flags: Dict[str, int | object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Convert to a JSON-serializable dictionary using semantic field names."""

        data: Dict[str, object] = {
            "opcode": self.opcode.name,
            "act": self.act.name if self.act is not None else None,
            "qmode": self.qmode.name if self.qmode is not None else None,
            "first_flag": self.first_flag,
            "last_flag": self.last_flag,
            "sync": self.sync,
            "h_tile": self.h_tile,
            "w_tile": self.w_tile,
            "c4_in": self.c4_in,
            "c4_out": self.c4_out,
            "y_index": self.y_index,
            "ifm_w": self.ifm_w,
            "ifm_h": self.ifm_h,
            "actdma_line_words": self.actdma_line_words,
            "outdma_line_words": self.outdma_line_words,
            "stride_h": self.stride_h,
            "stride_w": self.stride_w,
            "pad_top": self.pad_top,
            "pad_bottom": self.pad_bottom,
            "pad_left": self.pad_left,
            "pad_right": self.pad_right,
            "fi_stride": self.fi_stride,
            "fo_stride": self.fo_stride,
            "param_addr": self.param_addr,
            "fi_addr": self.fi_addr,
            "fo_addr": self.fo_addr,
            "flags": self.flags,
        }
        # Add hex string views for address-related fields to make JSON dumps easier to read.
        data["param_addr_hex"] = hex(self.param_addr)
        data["fi_addr_hex"] = hex(self.fi_addr)
        data["fo_addr_hex"] = hex(self.fo_addr)
        return data
