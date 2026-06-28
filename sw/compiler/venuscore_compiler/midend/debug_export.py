# -*- coding: utf-8 -*-
"""
Debug exporters for midend artifacts (LayerConfig and TilePlan).

This module is purely for human/debug consumption: it serializes the
LayerConfig / TileDesc information into a JSON snapshot that can be
consumed by testbenches, golden model scripts, or manual inspection.
"""

from __future__ import annotations

import json
from pathlib import Path

from venuscore_compiler.midend.types import LayerConfig, TileDesc, TilePlan
from venuscore_compiler.common.capacity import (
    WBUF_LANE_BYTES,
    WBUF_LANES,
    weight_bytes,
)

__all__ = ["export_tile_plan"]


def _layer_to_dict(cfg: LayerConfig) -> dict:
    return {
        "layer_id": getattr(cfg, "layer_id", -1),
        "name": cfg.name,
        "op_type": cfg.op_type,
        "ifm_h": cfg.ifm_h,
        "ifm_w": cfg.ifm_w,
        "ofm_h": cfg.ofm_h,
        "ofm_w": cfg.ofm_w,
        "cin": cfg.cin,
        "cout": cfg.cout,
        "c4_in": cfg.c4_in,
        "c4_out": cfg.c4_out,
        "stride_h": cfg.stride_h,
        "stride_w": cfg.stride_w,
        "pad_top": cfg.pad_top,
        "pad_bottom": cfg.pad_bottom,
        "pad_left": cfg.pad_left,
        "pad_right": cfg.pad_right,
        "fi_stride": cfg.fi_stride,
        "fo_stride": cfg.fo_stride,
        "layout": cfg.layout,
        "qmode": getattr(cfg, "qmode", "INT8"),
        "quant_table_size": len(getattr(cfg, "quant_table", [])),
        "kernel_h": getattr(cfg, "kernel_h", None),
        "kernel_w": getattr(cfg, "kernel_w", None),
    }


def _tile_to_dict(t: TileDesc) -> dict:
    # Estimate per-tile parameter bytes for easier capacity inspection.
    cout_tile = t.co_end - t.co_start
    quant_bytes = cout_tile * 8
    qmode = (t.metadata or {}).get("qmode", "INT8")
    kernel_bytes = weight_bytes(
        t.op_type,
        t.cin,
        cout_tile,
        t.kernel_h,
        t.kernel_w,
        qmode,
    )
    total_param = quant_bytes + kernel_bytes
    wbuf_capacity = WBUF_LANE_BYTES * WBUF_LANES

    return {
        "tile_index": t.tile_index,
        "layer_id": t.layer_id,
        "op_name": t.op_name,
        "op_type": t.op_type,
        "y_index": t.y_index,
        "h_tile": t.h_tile,
        "w_tile": t.w_tile,
        "h_in": t.h_in,
        "w_in": t.w_in,
        "pad_top": t.pad_top,
        "pad_bottom": t.pad_bottom,
        "pad_left": t.pad_left,
        "pad_right": t.pad_right,
        "cin": t.cin,
        "cout": t.cout,
        "co_start": t.co_start,
        "co_end": t.co_end,
        "cout_tile": cout_tile,
        "c4_in_start": t.c4_in_start,
        "c4_in_end": t.c4_in_end,
        "c4_out_start": t.c4_out_start,
        "c4_out_end": t.c4_out_end,
        "kernel_h": t.kernel_h,
        "kernel_w": t.kernel_w,
        "stride_h": t.stride_h,
        "stride_w": t.stride_w,
        "is_first_in_layer": t.is_first_in_layer,
        "is_last_in_layer": t.is_last_in_layer,
        "qmode": qmode,
        "total_param_bytes": total_param,
        "wbuf_capacity_bytes": wbuf_capacity,
    }


def export_tile_plan(tile_plan: TilePlan, path: str | Path) -> None:
    """
    Write a JSON snapshot of LayerConfig and TileDesc.

    JSON schema:

    {
      "layers": [ {LayerConfig...}, ... ],
      "tiles":  [ {TileDesc...}, ... ]
    }
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "layers": [_layer_to_dict(cfg) for cfg in tile_plan.layers],
        "tiles": [_tile_to_dict(t) for t in tile_plan.tiles],
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
