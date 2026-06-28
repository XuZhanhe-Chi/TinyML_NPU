# -*- coding: utf-8 -*-
"""
Hardware configuration descriptor for VenusCore NPU.

This module defines :class:`HwConfig`, which captures the hardware resource
limits and supported ISA feature subset for a concrete VenusCore target.
The values here are purely *capacity* / *capability* descriptions; functional
behavior is defined in the RTL and ISA documents.

All capacities are expressed in **bytes**, and opcode / qmode capabilities are
expressed using the ISA enums :class:`Opcode` and :class:`QMode`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set

from venuscore_compiler.isa.layout_spec import Opcode, QMode


@dataclass
class HwConfig:
    """Describes hardware limits and ISA feature subset for a VenusCore target.

    The compiler should treat this as the single source of truth for:

    - IBUF / WBUF capacity constraints (used by tiling / memory planning).
    - Supported ISA opcodes (which uOPs are legal to emit for this target).
    - Supported quantization modes (QMODEs) that can be executed in hardware.

    All size-related fields are in **bytes** unless explicitly documented
    otherwise.
    """

    # -------------------------------------------------------------------------
    # IBUF configuration
    # -------------------------------------------------------------------------
    # Maximum number of bytes that can be stored for a single IFM row of a tile
    # in IBUF. This is the host-side counterpart of IBUF_LINE_BYTES in the spec.
    ibuf_line_bytes: int = 3840

    # Optional IBUF geometry/capacity limits beyond a single row.
    # If set, the compiler will ensure a tile's required input height window
    # (h_in = h_tile * stride + kh - 1) fits within these limits.
    #
    # Leave as None when the hardware does not expose such a hard bound.
    ibuf_max_rows: int | None = None
    ibuf_total_bytes: int | None = None


    # -------------------------------------------------------------------------
    # WBUF configuration
    # -------------------------------------------------------------------------
    # Maximum bytes per WBUF lane. Total WBUF capacity is
    #   wbuf_lane_bytes * wbuf_lanes
    # and should match the RTL parameters.
    wbuf_lane_bytes: int = 1 << 16

    # Number of WBUF lanes that can be used in parallel (LANE_NUM).
    wbuf_lanes: int = 4

    # -------------------------------------------------------------------------
    # Geometric limits
    # -------------------------------------------------------------------------
    # Upper bound for H_TILE (tile output height) accepted by hardware.
    # The midend tiler must ensure 0 < H_TILE <= max_h_tile.
    max_h_tile: int = 255

    # Optional upper bound for output channels per tile, expressed in C4 groups.
    # When set, the tiler will cap Cout tiling so that:
    #   Cout_tile <= max_c4_out * 4
    #
    # This is useful when downstream micro-architecture (e.g. OBuf / DMA queue
    # depths) benefits from bounding the width of a single tile.
    max_c4_out: int | None = None

    # Number of clusters available in the NPU. Current pipeline assumes
    # a single cluster, but this field allows future multi-cluster tiling
    # or scheduling policies.
    cluster_count: int = 1

    # -------------------------------------------------------------------------
    # Compatibility knobs
    # -------------------------------------------------------------------------
    # Some historical RTL versions required IFM_H/IFM_W to be even for stride=2
    # spatial ops. Current compiler default relaxes this; flip to True if your
    # hardware still relies on that restriction.
    stride2_requires_even_ifm: bool = False

    # -------------------------------------------------------------------------
    # Capability flags
    # -------------------------------------------------------------------------
    # Subset of ISA opcodes that this hardware implementation actually supports.
    # This is expressed in terms of Opcode enums, not IR op_type strings.
    supported_opcodes: Set[Opcode] = field(
        default_factory=lambda: {
            Opcode.CONV2D,
            Opcode.PWCONV,
            Opcode.DWCONV,
            Opcode.AVGPOOL,
            Opcode.MAXPOOL,
        }
    )

    # Subset of QMODEs supported by this hardware.
    supported_qmodes: Set[QMode] = field(
        default_factory=lambda: {
            QMode.INT8,
        }
    )
