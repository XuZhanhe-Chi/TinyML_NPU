# -*- coding: utf-8 -*-
"""
Capacity and sizing helpers shared by midend and backend.

Defines:
  - IBUF_LINE_BYTES / IBUF_TOTAL_BYTES
  - WBUF_LANE_BYTES / WBUF_LANES
  - weight_bytes: compute weight storage size given op type, geometry and qmode.
"""

from __future__ import annotations

__all__ = [
    "IBUF_LINE_BYTES",
    "IBUF_TOTAL_BYTES",
    "WBUF_LANE_BYTES",
    "WBUF_LANES",
    "weight_bytes",
]

# Public ZYBO7010 preset. The compiler intentionally leaves 256 bytes of
# headroom below the RTL's physical 4096-byte IBUF line capacity.
IBUF_LINE_BYTES = 3840
IBUF_TOTAL_BYTES = 12 * 1024
WBUF_LANE_BYTES = 2048
WBUF_LANES = 4


def weight_bytes(op_type: str, cin: int, cout_tile: int, kh: int, kw: int, qmode: str) -> int:
    """Compute weight byte size for a tile based on op type and qmode."""

    bits_per_weight = 8
    if qmode == "INT4":
        bits_per_weight = 4
    elif qmode == "INT2":
        bits_per_weight = 2

    def packed_bytes(weight_count: int) -> int:
        return (weight_count * bits_per_weight + 7) // 8

    if op_type == "conv2d" or op_type == "pointwise_conv":
        # NOTE: Must match backend/layout_param_block.py (_emit_weights_conv2d):
        # - Weights are packed in 32-bit words, each word packs 4 int8 along Cin.
        # - Therefore storage is padded to C4 groups: c4_in = ceil(Cin/4).
        # This function is used by tiler/constraints to enforce WBUF capacity; it
        # must be conservative (never under-estimate) to avoid generating illegal tiles.

        cin_padded = ((cin + 3) // 4) * 4

        # For current VenusCore-v1, only INT8 is supported end-to-end.
        # Keep other qmode sizing conservative by applying the same C4 padding
        # and bit packing on the padded Cin.
        if qmode == "INT8":
            c4_in = cin_padded // 4
            return cout_tile * c4_in * kh * kw * 4

        return packed_bytes(cout_tile * cin_padded * kh * kw)
    if op_type == "depthwise_conv":
        # DWCONV is stored in a special packed format:
        #   per-channel, per-kw word: {top, mid, bot, 0} => 3 words / channel => 12B/channel
        # This must match docs/compiler.md and the backend parameter layout.
        return cout_tile * 3 * 4
    if op_type == "fully_connected":
        return packed_bytes(cout_tile * cin)
    if op_type in ("avg_pool", "avgpool", "max_pool", "maxpool"):
        return 0
    return packed_bytes(cout_tile * cin * kh * kw)
