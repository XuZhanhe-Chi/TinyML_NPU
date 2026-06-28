# -*- coding: utf-8 -*-
"""
Layout lowering: map logical tensor layouts to hardware-friendly logical layout info.

This pass stays hardware-agnostic in terms of *addresses*, but it:
  - Interprets tensor shapes under layout tags (NCHW / NHWC).
  - Produces per-op logical geometry for tiler:
      * IFM_H/W, OFM_H/W, Cin, Cout
      * C4_IN, C4_OUT
      * FI_STRIDE / FO_STRIDE (in 32-bit words, per global row)
  - Does not change the IR tensors themselves; it only returns a side-table.

Tiler consumes this side-table instead of directly inspecting raw tensor.shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from venuscore_compiler.ir.ops import (
    VcAvgPool,
    VcConv2D,
    VcDepthwiseConv,
    VcFullyConnected,
    VcMaxPool,
    VcOp,
    VcPointwiseConv,
)
from venuscore_compiler.ir.program import VcProgram

__all__ = ["LayerLayoutInfo", "lower_layouts"]


@dataclass
class LayerLayoutInfo:
    """Logical layout/geometry info for a single op (layer)."""

    name: str
    op_type: str
    ifm_h: int
    ifm_w: int
    ofm_h: int
    ofm_w: int
    cin: int
    cout: int
    c4_in: int
    c4_out: int
    fi_stride: int  # IFM global row stride in 32-bit words
    fo_stride: int  # OFM global row stride in 32-bit words
    layout: str     # logical layout tag, e.g., "NCHW"


def _as_nchw(shape: Tuple[int, int, int, int], layout: str) -> Tuple[int, int, int, int]:
    """Convert a (4D) shape from the given layout to (N,C,H,W)."""
    if len(shape) != 4:
        raise ValueError(f"Expected 4D shape, got {shape}.")
    n, d1, d2, d3 = shape
    layout = layout.upper()
    if layout == "NCHW":
        return n, d1, d2, d3
    if layout == "NHWC":
        return n, d3, d1, d2
    raise ValueError(f"Unsupported layout '{layout}'.")


def _infer_layout_info_for_op(op: VcOp, program: VcProgram) -> LayerLayoutInfo:
    if not op.inputs or not op.outputs:
        raise ValueError(f"[layout_lowering] Op '{op.name}' missing inputs/outputs.")
    try:
        ifm = program.tensors[op.inputs[0]]
        ofm = program.tensors[op.outputs[0]]
    except KeyError as e:  # pragma: no cover - defensive
        raise ValueError(
            f"[layout_lowering] Op '{op.name}' refers to unknown tensor '{e.args[0]}'."
        ) from e

    layout_in = (getattr(ifm, "layout", "NCHW") or "NCHW").upper()
    layout_out = (getattr(ofm, "layout", "NCHW") or "NCHW").upper()

    n_in, cin, ifm_h, ifm_w = _as_nchw(ifm.shape, layout_in)
    n_out, cout, ofm_h, ofm_w = _as_nchw(ofm.shape, layout_out)

    if n_in != 1 or n_out != 1:
        raise ValueError(
            f"[layout_lowering] Op '{op.name}' expects N==1 for feature maps "
            f"(got N_in={n_in}, N_out={n_out})."
        )

    c4_in = (cin + 3) // 4
    c4_out = (cout + 3) // 4

    # Logical FI/FO stride per ISA (word count per channel plane):
    #   FI_STRIDE = IFM_H * IFM_W
    #   FO_STRIDE = OFM_H * OFM_W
    # Channel packing is accounted for via C4 groups and addresses, not stride.
    fi_stride_words = ifm_h * ifm_w
    fo_stride_words = ofm_h * ofm_w

    # Early ISA field-range checks (avoid failing late during uOP encoding).
    # W6 packs IFM_W/IFM_H as 16-bit, and W2 packs FI_STRIDE/FO_STRIDE as 16-bit.
    if ifm_h > 0xFFFF or ifm_w > 0xFFFF:
        raise ValueError(
            f"[layout_lowering] Op '{op.name}' IFM dims out of range for ISA fields: "
            f"IFM_H={ifm_h}, IFM_W={ifm_w} (max 65535)."
        )
    if ofm_h > 0xFFFF or ofm_w > 0xFFFF:
        raise ValueError(
            f"[layout_lowering] Op '{op.name}' OFM dims out of range for ISA fields: "
            f"OFM_H={ofm_h}, OFM_W={ofm_w} (max 65535)."
        )
    if fi_stride_words > 0xFFFF or fo_stride_words > 0xFFFF:
        raise ValueError(
            f"[layout_lowering] Op '{op.name}' FI_STRIDE/FO_STRIDE out of range (16-bit): "
            f"FI_STRIDE={fi_stride_words}, FO_STRIDE={fo_stride_words}. "
            "This typically means IFM_H*IFM_W or OFM_H*OFM_W exceeds 65535."
        )

    return LayerLayoutInfo(
        name=op.name,
        op_type=getattr(op, "op_type", type(op).__name__.lower()),
        ifm_h=ifm_h,
        ifm_w=ifm_w,
        ofm_h=ofm_h,
        ofm_w=ofm_w,
        cin=cin,
        cout=cout,
        c4_in=c4_in,
        c4_out=c4_out,
        fi_stride=fi_stride_words,
        fo_stride=fo_stride_words,
        layout=layout_in,
    )


def lower_layouts(program: VcProgram) -> Dict[str, LayerLayoutInfo]:
    """
    Produce per-op LayerLayoutInfo side-table.

    Returns:
        dict: op_name -> LayerLayoutInfo
    """
    layout_info: Dict[str, LayerLayoutInfo] = {}

    for op in program.ops:
        # For now we only populate layout info for NPU ops; other ops keep it empty.
        if isinstance(
            op,
            (VcConv2D, VcDepthwiseConv, VcPointwiseConv, VcAvgPool, VcMaxPool, VcFullyConnected),
        ):
            info = _infer_layout_info_for_op(op, program)
            layout_info[op.name] = info

    return layout_info
