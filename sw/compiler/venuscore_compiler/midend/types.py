# -*- coding: utf-8 -*-
"""
Midend public data types for backend consumption.

This module centralizes the key logical structures produced by the midend:

- LayerConfig: per-layer geometry/stride/quantization metadata.
- TileDesc:    per-tile logical uOP view (no addresses).
- TilePlan:    container for all tiles and layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LayerConfig:
    """Layer-level configuration exposed to backend and memory planner."""

    layer_id: int
    name: str
    op_type: str
    act_type: str | None
    ifm_h: int
    ifm_w: int
    ofm_h: int
    ofm_w: int
    cin: int
    cout: int
    c4_in: int
    c4_out: int
    stride_h: int
    stride_w: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    fi_stride: int
    fo_stride: int
    # Tensor names for IFM/OFM binding
    ifm_name: str = ""
    ofm_name: str = ""
    input_name: str = ""
    output_name: str = ""
    weight_name: str = ""
    bias_name: str = ""
    # Layout / quantization
    logical_layout: str = "NCHW"
    physical_layout: str = "NCHWc4"
    qmode: str = "INT8"
    quant_table: list = field(default_factory=list)
    # Kernel info
    kernel_h: int = 3
    kernel_w: int = 3
    # Optional dtype annotations (activation/weight)
    activation_dtype: str = "int8"
    weight_dtype: str = "int8"


@dataclass
class TileDesc:
    """Logical uOP unit (tile) without addresses."""

    layer_id: int
    op_name: str
    op_type: str
    h_tile: int
    w_tile: int
    y_index: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    cin: int
    cout: int
    co_start: int
    co_end: int
    c4_in_start: int
    c4_in_end: int
    c4_out_start: int
    c4_out_end: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    h_in: int
    w_in: int
    act_type: str | None = None
    input_name: str = ""
    output_name: str = ""
    weight_name: str = ""
    bias_name: str = ""
    tile_id: int = 0
    tile_index: int = 0
    is_first_in_layer: bool = False
    is_last_in_layer: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cout_tile(self) -> int:
        return max(0, self.co_end - self.co_start)


@dataclass
class TilePlan:
    """Full tiling plan returned to backend / debug exporters."""

    tiles: List[TileDesc]
    tiles_by_op: Dict[str, List[TileDesc]]
    layers: List[LayerConfig]


__all__ = ["LayerConfig", "TileDesc", "TilePlan"]
