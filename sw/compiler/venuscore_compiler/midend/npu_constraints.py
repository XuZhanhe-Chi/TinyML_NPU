# -*- coding: utf-8 -*-
"""
Midend constraint checks for layer-level and tile-level validation.

Enforces:
  - Layer-level hardware subset:
      * Supported kernels/strides/padding/dilation per VenusCore ISA.
      * Cout alignment to C4 (multiple of 4).
      * QMODE within supported set.
  - Tile-level geometry/capacity:
      * No W-direction tiling (W_TILE == W_out).
      * No Cin tiling (Cin is full stripe, C4_IN covers all Cin).
      * IBUF line / total capacity.
      * WBUF parameter size per tile (weight + quant params).
"""

from __future__ import annotations

from typing import List, Dict

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
from venuscore_compiler.midend.types import TileDesc, TilePlan, LayerConfig
from venuscore_compiler.common.capacity import weight_bytes
from venuscore_compiler.config import HwConfig, default_hw_config

__all__ = ["check_layer_constraints", "check_tile_constraints"]

SUPPORTED_QMODES = {"INT8"}  # future: extend to {"INT8", "INT4", "INT2"}


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def _as_nchw(shape, layout: str):
    """Interpret a 4D shape under layout into (N, C, H, W)."""
    if len(shape) != 4:
        raise ValueError(f"Expected 4D shape, got {shape}.")
    n, d1, d2, d3 = shape
    layout = (layout or "").upper()
    if layout == "NCHW":
        return n, d1, d2, d3
    elif layout == "NHWC":
        return n, d3, d1, d2
    raise ValueError(f"Unsupported layout '{layout}'.")


def _get_io_tensors(op: VcOp, program: VcProgram):
    if not op.inputs or not op.outputs:
        raise ValueError(f"[constraints] Op '{op.name}' missing inputs/outputs.")
    try:
        inp = program.tensors[op.inputs[0]]
        out = program.tensors[op.outputs[0]]
    except KeyError as e:  # pragma: no cover - defensive
        raise ValueError(
            f"[constraints] Op '{op.name}' refers to unknown tensor '{e.args[0]}'."
        ) from e
    return inp, out


# ---------------------------------------------------------------------------
# Layer-level checks
# ---------------------------------------------------------------------------


def _check_conv_layer(op: VcConv2D, program: VcProgram, errors: List[str], target: str) -> None:
    inp, out = _get_io_tensors(op, program)
    n_in, cin, h_in, w_in = _as_nchw(inp.shape, inp.layout)
    n_out, cout, h_out, w_out = _as_nchw(out.shape, out.layout)

    if n_in != 1 or n_out != 1:
        errors.append(f"[{target}] Conv2D '{op.name}' requires N==1 (got N_in={n_in}, N_out={n_out}).")
    if cin <= 0 or cout <= 0:
        errors.append(f"[{target}] Conv2D '{op.name}' invalid Cin/Cout ({cin},{cout}).")

    kh, kw = op.kernel
    sh, sw = op.stride
    pt, pb, pl, pr = op.padding
    dh, dw = op.dilation

    # Hardware-supported kernels: 3x3 for Conv2D (1x1 is handled by VcPointwiseConv)
    if (kh, kw) != (3, 3):
        errors.append(
            f"[{target}] Conv2D '{op.name}' kernel {kh}x{kw} not supported (expect 3x3; "
            "use VcPointwiseConv for 1x1)."
        )
    # Stride: 1 or 2
    if sh not in (1, 2) or sw not in (1, 2):
        errors.append(
            f"[{target}] Conv2D '{op.name}' stride ({sh},{sw}) not supported (expect 1 or 2)."
        )
    # Padding per side: 0 or 1
    if pt not in (0, 1) or pb not in (0, 1) or pl not in (0, 1) or pr not in (0, 1):
        errors.append(
            f"[{target}] Conv2D '{op.name}' padding ({pt},{pb},{pl},{pr}) not supported "
            "(only 0 or 1 per side)."
        )
    # Dilation: only 1 supported
    if (dh, dw) != (1, 1):
        errors.append(
            f"[{target}] Conv2D '{op.name}' dilation ({dh},{dw}) not supported (expect 1,1)."
        )

    # Output shape sanity against standard conv formula
    h_eff = h_in + pt + pb
    w_eff = w_in + pl + pr
    h_exp = (h_eff - kh) // sh + 1
    w_exp = (w_eff - kw) // sw + 1
    if h_exp != h_out or w_exp != w_out:
        errors.append(
            f"[{target}] Conv2D '{op.name}' output shape mismatch: expected ({h_exp},{w_exp}), "
            f"got ({h_out},{w_out})."
        )

    # Cout alignment: multiple of 4 (C4 pack)
    if cout % 4 != 0:
        errors.append(
            f"[{target}] Conv2D '{op.name}' Cout={cout} not multiple of 4; "
            "VenusCore requires C4 alignment."
        )

    # QMODE support
    qmode = getattr(op, "qmode", None)
    if qmode is not None and qmode not in SUPPORTED_QMODES:
        errors.append(
            f"[{target}] Conv2D '{op.name}' uses unsupported qmode='{qmode}'. "
            f"Supported: {sorted(SUPPORTED_QMODES)}."
        )


def _check_depthwise_layer(op: VcDepthwiseConv, program: VcProgram, errors: List[str], target: str) -> None:
    inp, out = _get_io_tensors(op, program)
    n_in, cin, h_in, w_in = _as_nchw(inp.shape, inp.layout)
    n_out, cout, h_out, w_out = _as_nchw(out.shape, out.layout)

    if n_in != 1 or n_out != 1:
        errors.append(f"[{target}] DepthwiseConv '{op.name}' requires N==1.")
    if cin <= 0 or cout <= 0 or cin != cout:
        errors.append(
            f"[{target}] DepthwiseConv '{op.name}' expects Cin==Cout>0, "
            f"got Cin={cin}, Cout={cout}."
        )

    kh, kw = op.kernel
    sh, sw = op.stride
    pt, pb, pl, pr = op.padding
    dh, dw = op.dilation

    if (kh, kw) != (3, 3):
        errors.append(
            f"[{target}] DepthwiseConv '{op.name}' kernel {kh}x{kw} not supported (expect 3x3)."
        )
    if sh not in (1, 2) or sw not in (1, 2):
        errors.append(
            f"[{target}] DepthwiseConv '{op.name}' stride ({sh},{sw}) not supported (expect 1 or 2)."
        )
    if pt not in (0, 1) or pb not in (0, 1) or pl not in (0, 1) or pr not in (0, 1):
        errors.append(
            f"[{target}] DepthwiseConv '{op.name}' padding ({pt},{pb},{pl},{pr}) not supported "
            "(only 0 or 1 per side)."
        )
    if (dh, dw) != (1, 1):
        errors.append(
            f"[{target}] DepthwiseConv '{op.name}' dilation ({dh},{dw}) not supported (expect 1,1)."
        )

    h_eff = h_in + pt + pb
    w_eff = w_in + pl + pr
    h_exp = (h_eff - kh) // sh + 1
    w_exp = (w_eff - kw) // sw + 1
    if h_exp != h_out or w_exp != w_out:
        errors.append(
            f"[{target}] DepthwiseConv '{op.name}' output shape mismatch: "
            f"expected ({h_exp},{w_exp}), got ({h_out},{w_out})."
        )

    if cout % 4 != 0:
        errors.append(
            f"[{target}] DepthwiseConv '{op.name}' Cout={cout} not multiple of 4; "
            "VenusCore requires C4 alignment."
        )

    qmode = getattr(op, "qmode", None)
    if qmode is not None and qmode not in SUPPORTED_QMODES:
        errors.append(
            f"[{target}] DepthwiseConv '{op.name}' uses unsupported qmode='{qmode}'. "
            f"Supported: {sorted(SUPPORTED_QMODES)}."
        )


def _check_pointwise_layer(op: VcPointwiseConv, program: VcProgram, errors: List[str], target: str) -> None:
    inp, out = _get_io_tensors(op, program)
    n_in, cin, h_in, w_in = _as_nchw(inp.shape, inp.layout)
    n_out, cout, h_out, w_out = _as_nchw(out.shape, out.layout)

    if n_in != 1 or n_out != 1:
        errors.append(f"[{target}] PointwiseConv '{op.name}' requires N==1.")
    if cin <= 0 or cout <= 0:
        errors.append(f"[{target}] PointwiseConv '{op.name}' invalid Cin/Cout ({cin},{cout}).")

    kh, kw = op.kernel
    sh, sw = op.stride
    pt, pb, pl, pr = op.padding
    dh, dw = op.dilation

    if (kh, kw) != (1, 1):
        errors.append(
            f"[{target}] PointwiseConv '{op.name}' kernel {kh}x{kw} not supported (expect 1x1)."
        )
    if sh not in (1, 2) or sw not in (1, 2):
        errors.append(
            f"[{target}] PointwiseConv '{op.name}' stride ({sh},{sw}) not supported (expect 1 or 2)."
        )
    if pt not in (0, 1) or pb not in (0, 1) or pl not in (0, 1) or pr not in (0, 1):
        errors.append(
            f"[{target}] PointwiseConv '{op.name}' padding ({pt},{pb},{pl},{pr}) not supported "
            "(only 0 or 1 per side)."
        )
    if (dh, dw) != (1, 1):
        errors.append(
            f"[{target}] PointwiseConv '{op.name}' dilation ({dh},{dw}) not supported (expect 1,1)."
        )

    # Output shape sanity: effectively 1x1 conv
    h_eff = h_in + pt + pb
    w_eff = w_in + pl + pr
    h_exp = (h_eff - kh) // sh + 1
    w_exp = (w_eff - kw) // sw + 1
    if h_exp != h_out or w_exp != w_out:
        errors.append(
            f"[{target}] PointwiseConv '{op.name}' output shape mismatch: "
            f"expected ({h_exp},{w_exp}), got ({h_out},{w_out})."
        )

    if cout % 4 != 0:
        errors.append(
            f"[{target}] PointwiseConv '{op.name}' Cout={cout} not multiple of 4; "
            "VenusCore requires C4 alignment."
        )

    qmode = getattr(op, "qmode", None)
    if qmode is not None and qmode not in SUPPORTED_QMODES:
        errors.append(
            f"[{target}] PointwiseConv '{op.name}' uses unsupported qmode='{qmode}'. "
            f"Supported: {sorted(SUPPORTED_QMODES)}."
        )


def _check_avgpool_layer(op: VcAvgPool, program: VcProgram, errors: List[str], target: str) -> None:
    inp, out = _get_io_tensors(op, program)
    n_in, cin, h_in, w_in = _as_nchw(inp.shape, inp.layout)
    n_out, cout, h_out, w_out = _as_nchw(out.shape, out.layout)

    if n_in != 1 or n_out != 1:
        errors.append(f"[{target}] AvgPool '{op.name}' requires N==1.")
    if cin != cout:
        errors.append(f"[{target}] AvgPool '{op.name}' expects Cin==Cout, got {cin},{cout}.")

    kh, kw = op.kernel
    sh, sw = op.stride
    pt, pb, pl, pr = op.padding

    if kh != 2 or kw != 2:
        errors.append(f"[{target}] AvgPool '{op.name}' kernel {kh}x{kw} not supported (expect 2x2).")
    if sh not in (1, 2) or sw not in (1, 2):
        errors.append(
            f"[{target}] AvgPool '{op.name}' stride ({sh},{sw}) not supported (expect 1 or 2)."
        )
    if pt not in (0, 1) or pb not in (0, 1) or pl not in (0, 1) or pr not in (0, 1):
        errors.append(
            f"[{target}] AvgPool '{op.name}' padding ({pt},{pb},{pl},{pr}) not supported "
            "(only 0 or 1 per side)."
        )

    h_eff = h_in + pt + pb
    w_eff = w_in + pl + pr
    h_exp = (h_eff - kh) // sh + 1
    w_exp = (w_eff - kw) // sw + 1
    if h_exp != h_out or w_exp != w_out:
        errors.append(
            f"[{target}] AvgPool '{op.name}' output shape mismatch: "
            f"expected ({h_exp},{w_exp}), got ({h_out},{w_out})."
        )


def _check_maxpool_layer(op: VcMaxPool, program: VcProgram, errors: List[str], target: str) -> None:
    inp, out = _get_io_tensors(op, program)
    n_in, cin, h_in, w_in = _as_nchw(inp.shape, inp.layout)
    n_out, cout, h_out, w_out = _as_nchw(out.shape, out.layout)

    if n_in != 1 or n_out != 1:
        errors.append(f"[{target}] MaxPool '{op.name}' requires N==1.")
    if cin != cout:
        errors.append(f"[{target}] MaxPool '{op.name}' expects Cin==Cout, got {cin},{cout}.")

    kh, kw = op.kernel
    sh, sw = op.stride
    pt, pb, pl, pr = op.padding

    if kh != 2 or kw != 2:
        errors.append(f"[{target}] MaxPool '{op.name}' kernel {kh}x{kw} not supported (expect 2x2).")
    if sh != 2 or sw != 2:
        errors.append(
            f"[{target}] MaxPool '{op.name}' stride ({sh},{sw}) not supported (expect 2)."
        )
    if pt != 0 or pb != 0 or pl != 0 or pr != 0:
        errors.append(
            f"[{target}] MaxPool '{op.name}' padding ({pt},{pb},{pl},{pr}) not supported "
            "(all must be 0)."
        )

    h_eff = h_in + pt + pb
    w_eff = w_in + pl + pr
    h_exp = (h_eff - kh) // sh + 1
    w_exp = (w_eff - kw) // sw + 1
    if h_exp != h_out or w_exp != w_out:
        errors.append(
            f"[{target}] MaxPool '{op.name}' output shape mismatch: "
            f"expected ({h_exp},{w_exp}), got ({h_out},{w_out})."
        )


def _check_fc_layer(op: VcFullyConnected, program: VcProgram, errors: List[str], target: str) -> None:
    inp, out = _get_io_tensors(op, program)
    n_in, cin, h_in, w_in = _as_nchw(inp.shape, inp.layout)
    n_out, cout, h_out, w_out = _as_nchw(out.shape, out.layout)

    if n_in != 1 or n_out != 1:
        errors.append(f"[{target}] FullyConnected '{op.name}' requires N==1.")
    if h_in != 1 or w_in != 1 or h_out != 1 or w_out != 1:
        errors.append(
            f"[{target}] FullyConnected '{op.name}' expects H=W=1 for input/output "
            f"(got H_in={h_in}, W_in={w_in}, H_out={h_out}, W_out={w_out})."
        )
    if cin <= 0 or cout <= 0:
        errors.append(f"[{target}] FullyConnected '{op.name}' invalid Cin/Cout ({cin},{cout}).")

    if cout % 4 != 0:
        errors.append(
            f"[{target}] FullyConnected '{op.name}' Cout={cout} not multiple of 4; "
            "VenusCore requires C4 alignment."
        )

    qmode = getattr(op, "qmode", None)
    if qmode is not None and qmode not in SUPPORTED_QMODES:
        errors.append(
            f"[{target}] FullyConnected '{op.name}' uses unsupported qmode='{qmode}'. "
            f"Supported: {sorted(SUPPORTED_QMODES)}."
        )


def check_layer_constraints(program: VcProgram, target: str = "venuscore-v1") -> None:
    """
    Run layer-level hardware capability checks on the logical program.

    This should be called after IR normalization and before tiling.
    """
    errors: List[str] = []

    for op in program.ops:
        if isinstance(op, VcConv2D):
            _check_conv_layer(op, program, errors, target)
        elif isinstance(op, VcDepthwiseConv):
            _check_depthwise_layer(op, program, errors, target)
        elif isinstance(op, VcPointwiseConv):
            _check_pointwise_layer(op, program, errors, target)
        elif isinstance(op, VcAvgPool):
            _check_avgpool_layer(op, program, errors, target)
        elif isinstance(op, VcMaxPool):
            _check_maxpool_layer(op, program, errors, target)
        elif isinstance(op, VcFullyConnected):
            _check_fc_layer(op, program, errors, target)
        else:
            # Non-NPU ops: ignored here.
            continue

    if errors:
        # Aggregate all errors and raise once, so frontend/tests can see everything together.
        raise ValueError("\n".join(errors))


# ---------------------------------------------------------------------------
# Tile-level checks
# ---------------------------------------------------------------------------


def check_tile_constraints(
    tile_plan: TilePlan,
    program: VcProgram,
    target: str = "venuscore-v1",
    hw: HwConfig | None = None,
) -> None:
    """
    Run tile-level capacity and geometry checks.

    Enforces:
      - No W tiling: each tile covers full W_out.
      - No Cin tiling: Cin and C4_IN cover full input channels.
      - IBUF line/total capacity.
      - WBUF capacity per tile (kernel + quant params).
    """
    errors: List[str] = []
    if hw is None:
        hw = default_hw_config()

    # Build map layer_id -> LayerConfig for quick lookup.
    layers_by_id: Dict[int, LayerConfig] = {}
    for cfg in tile_plan.layers:
        layer_id = getattr(cfg, "layer_id", None)
        if layer_id is None:
            continue
        layers_by_id[layer_id] = cfg

    total_wbuf_bytes = hw.wbuf_lane_bytes * hw.wbuf_lanes

    for tile in tile_plan.tiles:
        cfg = layers_by_id.get(tile.layer_id)
        if cfg is None:
            errors.append(
                f"[{target}] Tile '{tile.op_name}' has unknown layer_id={tile.layer_id}."
            )
            continue

        # 1) No W-direction tiling: W_TILE must equal OFM_W.
        if tile.w_tile != cfg.ofm_w:
            errors.append(
                f"[{target}] Tile {tile.op_name}#{tile.tile_index}: W_TILE={tile.w_tile} "
                f"!= OFM_W={cfg.ofm_w} (W tiling is not supported)."
            )

        # 2) Cin tiling:
        # v1 generally forbids Cin tiling, but channel-wise ops may need channel slicing.
        # - AvgPool: no weights, but quant coeff bytes (Cout_tile*8) must fit into WBUF.
        # - DWConv: Cin==Cout and channel tiling implies the same slice on input.
        if tile.op_type in ("avg_pool", "avgpool", "max_pool", "maxpool", "depthwise_conv", "dwconv"):
            if tile.cin != tile.cout:
                errors.append(
                    f"[{target}] Channel-wise tile {tile.op_name}#{tile.tile_index}: "
                    f"Cin={tile.cin} != Cout_tile={tile.cout}."
                )
            if tile.c4_in_start != tile.c4_out_start or tile.c4_in_end != tile.c4_out_end:
                errors.append(
                    f"[{target}] Channel-wise tile {tile.op_name}#{tile.tile_index}: C4_IN range "
                    f"[{tile.c4_in_start},{tile.c4_in_end}) must match C4_OUT range "
                    f"[{tile.c4_out_start},{tile.c4_out_end})."
                )
        else:
            if tile.cin != cfg.cin:
                errors.append(
                    f"[{target}] Tile {tile.op_name}#{tile.tile_index}: Cin={tile.cin} != layer Cin={cfg.cin} "
                    "(Cin tiling is not supported)."
                )
            if tile.c4_in_start != 0 or tile.c4_in_end != cfg.c4_in:
                errors.append(
                    f"[{target}] Tile {tile.op_name}#{tile.tile_index}: C4_IN range "
                    f"[{tile.c4_in_start},{tile.c4_in_end}) != [0,{cfg.c4_in}) "
                    "(C4_IN must cover full Cin)."
                )

        # 3) IBUF line capacity: cin * w_in <= ibuf_line_bytes.
        bytes_per_row = tile.cin * tile.w_in
        if bytes_per_row > hw.ibuf_line_bytes:
            errors.append(
                f"[{target}] Tile {tile.op_name}#{tile.tile_index}: "
                f"cin * w_in = {tile.cin} * {tile.w_in} = {bytes_per_row} "
                f"> ibuf_line_bytes={hw.ibuf_line_bytes}."
            )

        # Optional IBUF limits beyond single-line bytes.
        if getattr(hw, "ibuf_max_rows", None) is not None:
            if tile.h_in > int(hw.ibuf_max_rows):
                errors.append(
                    f"[{target}] Tile {tile.op_name}#{tile.tile_index}: "
                    f"h_in={tile.h_in} > ibuf_max_rows={hw.ibuf_max_rows}."
                )
        if getattr(hw, "ibuf_total_bytes", None) is not None:
            bytes_total = bytes_per_row * tile.h_in
            if bytes_total > int(hw.ibuf_total_bytes):
                errors.append(
                    f"[{target}] Tile {tile.op_name}#{tile.tile_index}: "
                    f"ibuf bytes needed={bytes_total} > ibuf_total_bytes={hw.ibuf_total_bytes}."
                )

        # 4) WBUF capacity: weight params + quant params per tile.
        cout_tile = tile.co_end - tile.co_start
        quant_bytes = cout_tile * 8  # bias_s32(4B) + mul_u16(2B) + shift_u6/pad(2B) ≈ 8B/OC
        qmode = (tile.metadata or {}).get("qmode", "INT8")

        kernel_bytes = weight_bytes(
            tile.op_type,
            tile.cin,
            cout_tile,
            tile.kernel_h,
            tile.kernel_w,
            qmode,
        )
        total_param = quant_bytes + kernel_bytes
        if total_param > total_wbuf_bytes:
            errors.append(
                f"[{target}] Tile {tile.op_name}#{tile.tile_index}: param bytes "
                f"{total_param} > WBUF capacity {total_wbuf_bytes}."
            )

    if errors:
        raise ValueError("\n".join(errors))
