# -*- coding: utf-8 -*-
"""
Midend tiler: produce TilePlan and LayerConfig under VenusCore constraints.

Constraints:
  - H + Cout tiling only:
      * H: split along output height via (Y_INDEX, H_TILE).
      * W: no tiling, W_TILE == W_out.
      * Cout: split along output channels, aligned to C4 groups.
  - No Cin tiling: Cin is always the full stripe for the layer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from venuscore_compiler.ir.ops import (
    VcAvgPool,
    VcConv2D,
    VcDepthwiseConv,
    VcMaxPool,
    VcOp,
    VcPointwiseConv,
)
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.common.capacity import weight_bytes
from venuscore_compiler.config import HwConfig, default_hw_config
from venuscore_compiler.midend.layout_lowering import LayerLayoutInfo
from venuscore_compiler.midend.types import LayerConfig, TileDesc, TilePlan

__all__ = ["tile_program"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _OpTilingOverride:
    name_re: re.Pattern[str]
    force_divisible: bool | None = None
    allow_wide_pw: bool = False
    ignore_max_c4_out: bool = False
    max_cout_tile: int | None = None
    max_extra_stripes: int | None = None


_OP_TILING_OVERRIDES: list[_OpTilingOverride] = [
    # FC-like (AD): 1x1 PWCONV 1x1，优先减少切片数量（减少调度开销），再由 scheduler 做 2c 分配。
    _OpTilingOverride(
        name_re=re.compile(r"^/fcs\.\d+/Gemm$"),
        force_divisible=False,
        allow_wide_pw=True,
        ignore_max_c4_out=True,
        max_extra_stripes=1,
    ),
]


def _match_op_override(op: VcOp) -> _OpTilingOverride | None:
    name = op.name or ""
    for ov in _OP_TILING_OVERRIDES:
        if ov.name_re.search(name):
            return ov
    return None


def _schedule_tiles_for_clusters(tiles: List["TileDesc"], cluster_count: int) -> List["TileDesc"]:
    """
    Reorder tiles to improve multi-cluster utilization.

    Current policy:
      - Prefer alternating Y stripes early, so the first ``cluster_count`` uOPs
        are likely to target different H stripes (when available).
      - Keep Cout stripe order stable within each Y stripe.

    This is a best-effort heuristic; it must preserve correctness (no W tiling,
    disjoint OFM writes per tile).
    """
    if cluster_count <= 1 or len(tiles) <= 1:
        return tiles

    buckets: Dict[int, List["TileDesc"]] = {}
    for t in tiles:
        buckets.setdefault(int(t.y_index), []).append(t)

    y_keys = sorted(buckets.keys())
    for y in y_keys:
        buckets[y].sort(key=lambda t: int(t.co_start))

    scheduled: List["TileDesc"] = []
    i = 0
    while True:
        progressed = False
        for y in y_keys:
            if i < len(buckets[y]):
                scheduled.append(buckets[y][i])
                progressed = True
        if not progressed:
            break
        i += 1

    if len(scheduled) != len(tiles):
        # Defensive fallback: preserve original order.
        return tiles
    return scheduled


def _compute_cout_stripes(
    op_type: str,
    cin: int,
    cout: int,
    ifm_h: int,
    ifm_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    qmode: str,
    hw: HwConfig,
    cluster_count: int,
    h_stripe_count: int,
    max_cout_tile: int | None = None,
    force_divisible: bool | None = None,
    allow_wide_pw: bool = False,
    ignore_max_c4_out: bool = False,
    max_extra_stripes: int | None = None,
) -> List[Tuple[int, int]]:
    """
    Split Cout into stripes under WBUF capacity, aiming for balanced stripes and
    trying to make tile_count divisible by cluster_count when combined with H stripes.
    """
    stripes: List[Tuple[int, int]] = []
    remaining = cout
    co_start = 0

    total_wbuf_bytes = hw.wbuf_lane_bytes * hw.wbuf_lanes

    def _fits(cout_tile: int) -> bool:
        quant_bytes = cout_tile * 8
        kernel_bytes = weight_bytes(op_type, cin, cout_tile, kernel_h, kernel_w, qmode)
        return quant_bytes + kernel_bytes <= total_wbuf_bytes

    # Find maximum feasible C4-aligned stripe.
    max_feasible = remaining
    if max_cout_tile is not None:
        max_feasible = min(max_feasible, int(max_cout_tile))
    if hw.max_c4_out is not None and not ignore_max_c4_out:
        max_feasible = min(max_feasible, int(hw.max_c4_out) * 4)
    # Workaround: FC-like PWCONV (Cin small) is prone to output-drain pressure on
    # integrations with small OBuf / shared DMA. Limit Cout tiling more aggressively.
    if not allow_wide_pw and op_type == "pointwise_conv" and cin <= 8:
        max_feasible = min(max_feasible, 32)
    max_feasible = (max_feasible // 4) * 4
    while max_feasible > 0 and not _fits(max_feasible):
        max_feasible -= 4
    if max_feasible <= 0:
        raise ValueError(
            f"[tiler] Cannot fit any Cout stripe into WBUF (op_type={op_type}, "
            f"Cin={cin}, Cout={cout}, kernel={kernel_h}x{kernel_w}, qmode={qmode})."
        )

    while remaining > 0:
        stripes_left = max(1, math.ceil(remaining / max_feasible))
        stripe_size = math.ceil(remaining / stripes_left)
        # Align to C4 and cap by max_feasible.
        stripe_size = (stripe_size + 3) // 4 * 4
        stripe_size = min(stripe_size, max_feasible)
        # Ensure capacity fits; shrink if necessary.
        while stripe_size > 0 and not _fits(stripe_size):
            stripe_size -= 4
        if stripe_size <= 0:
            raise ValueError(
                f"[tiler] Cannot allocate Cout stripe under WBUF capacity "
                f"(remaining={remaining}, max_feasible={max_feasible})."
            )
        co_end = co_start + stripe_size
        stripes.append((co_start, co_end))
        remaining -= stripe_size
        co_start = co_end

    def _is_fc_like() -> bool:
        if op_type not in ("pointwise_conv", "conv2d"):
            return False
        if (kernel_h, kernel_w) != (1, 1):
            return False
        if (stride_h, stride_w) != (1, 1):
            return False
        return ifm_h == 1 and ifm_w == 1

    prefer_divisible = (
        cluster_count > 1
        and op_type in ("conv2d", "pointwise_conv", "depthwise_conv", "dwconv")
        and not _is_fc_like()
        and (ifm_h * ifm_w) >= 4
    )
    if force_divisible is not None:
        prefer_divisible = force_divisible

    # Multi-cluster policy:
    # - Ensure we have at least ``cluster_count`` stripes so clusters can run concurrently.
    # - Prefer divisibility for spatial conv layers to reduce tail effects in 2c,
    #   but keep it off for FC-like 1x1 workloads to avoid uOP explosion.
    total_tiles = len(stripes) * max(1, h_stripe_count)
    if cluster_count > 1 and h_stripe_count > 0:
        if total_tiles < cluster_count:
            stripes = _split_stripes_for_cluster(stripes, _fits, cluster_count)
        elif prefer_divisible and (total_tiles % cluster_count != 0):
            stripes_div = _split_stripes_for_divisibility(
                stripes,
                _fits,
                cluster_count,
                h_stripe_count,
            )
            extra_limit = max_extra_stripes if max_extra_stripes is not None else max(1, cluster_count)
            if len(stripes_div) <= len(stripes) + extra_limit:
                stripes = stripes_div

    return stripes


def _split_stripes_for_cluster(
    stripes: List[Tuple[int, int]],
    fits_fn,
    cluster_count: int,
) -> List[Tuple[int, int]]:
    """Attempt to increase stripe count (C4-aligned) to approach cluster_count."""

    stripes = list(stripes)
    while len(stripes) < cluster_count:
        # Find the widest stripe
        idx = max(range(len(stripes)), key=lambda i: stripes[i][1] - stripes[i][0])
        start, end = stripes[idx]
        size = end - start
        if size <= 4:
            break
        # Split roughly in half, align to C4
        mid = ((size // 2 + 3) // 4) * 4
        if mid >= size or mid <= 0:
            break
        left = (start, start + mid)
        right = (start + mid, end)
        if not fits_fn(left[1] - left[0]) or not fits_fn(right[1] - right[0]):
            break
        stripes.pop(idx)
        stripes.insert(idx, right)
        stripes.insert(idx, left)
    return stripes


def _split_stripes_for_divisibility(
    stripes: List[Tuple[int, int]],
    fits_fn,
    cluster_count: int,
    h_stripe_count: int,
) -> List[Tuple[int, int]]:
    """
    Attempt to split Cout stripes so that total tile count (H stripes * Cout stripes)
    becomes divisible by cluster_count. Uses a greedy largest-stripe split while
    respecting C4 alignment and WBUF capacity.
    """
    if cluster_count <= 1:
        return stripes
    stripes = list(stripes)
    while (len(stripes) * h_stripe_count) % cluster_count != 0:
        idx = max(range(len(stripes)), key=lambda i: stripes[i][1] - stripes[i][0])
        start, end = stripes[idx]
        size = end - start
        if size <= 4:
            break
        # Try to split roughly in half with C4 alignment.
        mid = ((size // 2 + 3) // 4) * 4
        if mid <= 0 or mid >= size:
            break
        left_size = mid
        right_size = size - mid
        if not fits_fn(left_size) or not fits_fn(right_size):
            # If the half split does not fit, try shrinking the split point.
            success = False
            candidate = mid - 4
            while candidate > 0:
                if fits_fn(candidate) and fits_fn(size - candidate):
                    mid = candidate
                    left_size = candidate
                    right_size = size - candidate
                    success = True
                    break
                candidate -= 4
            if not success:
                break
        left = (start, start + left_size)
        right = (start + left_size, end)
        stripes.pop(idx)
        stripes.insert(idx, right)
        stripes.insert(idx, left)
    return stripes


def _compute_h_stripes(
    ifm_h: int,
    ifm_w: int,
    cin: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    hw: HwConfig,
    cluster_count: int,
    op_type: str,
    cout: int,
    qmode: str,
    max_cout_tile: int | None = None,
    force_divisible: bool | None = None,
    allow_wide_pw: bool = False,
    ignore_max_c4_out: bool = False,
    max_extra_stripes: int | None = None,
) -> List[Tuple[int, int]]:
    """
    Compute H stripes as pairs (y_index, h_tile).

    Strategy:
      - Enumerate a small set of feasible H stripe counts.
      - For each candidate, estimate a simple "cost" (bytes moved + per-tile overhead),
        using the current Cout tiling under WBUF capacity.
      - Pick the lowest-cost candidate (tie-break: fewer stripes).

    Notes:
      - This is a performance heuristic; correctness constraints are still enforced
        by check_layer_constraints/check_tile_constraints.
      - Weight is reloaded from PARAM_ADDR per tile on current hardware, therefore
        extra H stripes multiply parameter traffic; the heuristic penalizes that.
    """
    stripes: List[Tuple[int, int]] = []

    h_eff = ifm_h + pad_top + pad_bottom
    h_out = (h_eff - kernel_h) // stride_h + 1

    if h_out <= 0:
        raise ValueError(
            f"[tiler] Invalid derived H_out={h_out} (IFM_H={ifm_h}, pad_top={pad_top}, "
            f"pad_bottom={pad_bottom}, kernel_h={kernel_h}, stride_h={stride_h})."
        )

    # ---------------------------------------------------------------------
    # Candidate H stripe counts
    # ---------------------------------------------------------------------
    min_stripes = max(1, math.ceil(h_out / hw.max_h_tile))
    # Search window: small, but includes cluster-oriented options when possible.
    max_search = min(h_out, max(min_stripes, min(cluster_count * 4, 32)))

    def _build_stripes(stripe_count: int) -> List[Tuple[int, int]]:
        if stripe_count <= 0:
            return []
        if math.ceil(h_out / stripe_count) > hw.max_h_tile:
            return []
        out: List[Tuple[int, int]] = []
        remaining = h_out
        y = 0
        for i in range(stripe_count):
            stripes_left = stripe_count - i
            h_tile = math.ceil(remaining / stripes_left)
            h_tile = min(h_tile, hw.max_h_tile)
            if h_tile <= 0:
                return []
            out.append((y, h_tile))
            remaining -= h_tile
            y += h_tile
        if remaining != 0:
            # Should not happen, but keep defensive.
            return []
        return out

    # ---------------------------------------------------------------------
    # Cost model (bytes proxy)
    # ---------------------------------------------------------------------
    # Per-tile fixed overhead (rough control + DMA setup); tuned per op type.
    overhead_per_tile = {
        "avg_pool": 1024,
        "avgpool": 1024,
        "max_pool": 1024,
        "maxpool": 1024,
        "depthwise_conv": 512,
        "dwconv": 512,
        "conv2d": 256,
        "pointwise_conv": 256,
    }.get(op_type, 256)

    # W tiling is forbidden by ISA; keep w_tile full.
    w_tile = (ifm_w + pad_left + pad_right - kernel_w) // stride_w + 1

    best_score: float | None = None
    best_cost: int | None = None
    best_stripes: List[Tuple[int, int]] | None = None
    best_tile_count: int | None = None

    def _estimate_parallelism(tile_count: int) -> float:
        """
        Heuristic effective parallelism for multi-cluster execution.

        Notes:
          - In reality, DMA is shared and speedup will be sub-linear; this is a
            tiler-side bias to ensure we generate enough tiles to keep clusters busy.
          - Penalize non-divisible tile counts slightly to reflect tail effects.
        """
        if cluster_count <= 1 or tile_count <= 0:
            return 1.0
        if tile_count >= cluster_count:
            # Mild tail penalty when tile_count is not divisible by cluster_count.
            rem = tile_count % cluster_count
            if rem == 0:
                return float(cluster_count)
            tail_penalty = 1.0 - 0.25 * (rem / float(tile_count))
            return float(cluster_count) * max(0.5, tail_penalty)
        return float(tile_count)

    for stripe_count in range(min_stripes, max_search + 1):
        candidate = _build_stripes(stripe_count)
        if not candidate:
            continue

        # Derive Cout stripes under WBUF capacity (depends on h_stripe_count for cluster tuning).
        cout_stripes = _compute_cout_stripes(
            op_type=op_type,
            cin=cin,
            cout=cout,
            ifm_h=ifm_h,
            ifm_w=ifm_w,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
            qmode=qmode,
            hw=hw,
            cluster_count=cluster_count,
            h_stripe_count=stripe_count,
            max_cout_tile=max_cout_tile,
            force_divisible=force_divisible,
            allow_wide_pw=allow_wide_pw,
            ignore_max_c4_out=ignore_max_c4_out,
            max_extra_stripes=max_extra_stripes,
        )

        tile_count = stripe_count * len(cout_stripes)
        cost = tile_count * overhead_per_tile
        invalid = False

        # Estimate bytes per tile; include halo overlap via h_in_tile.
        for _, h_tile in candidate:
            h_in_tile = h_tile * stride_h + kernel_h - 1
            for co_start, co_end in cout_stripes:
                cout_tile = co_end - co_start
                if op_type in ("avg_pool", "avgpool", "max_pool", "maxpool", "depthwise_conv", "dwconv"):
                    cin_tile = cout_tile
                else:
                    cin_tile = cin

                # Optional IBUF limits (beyond single-line bytes).
                if hw.ibuf_max_rows is not None and h_in_tile > hw.ibuf_max_rows:
                    invalid = True
                    break
                if hw.ibuf_total_bytes is not None:
                    bytes_needed = h_in_tile * ifm_w * cin_tile
                    if bytes_needed > hw.ibuf_total_bytes:
                        invalid = True
                        break

                # Activation traffic (NCHWc4 int8): bytes are element count.
                act_in_bytes = h_in_tile * ifm_w * cin_tile
                act_out_bytes = h_tile * w_tile * cout_tile

                # Param traffic: quant + weight, per tile.
                quant_bytes = cout_tile * 8
                kernel_bytes = weight_bytes(op_type, cin_tile, cout_tile, kernel_h, kernel_w, qmode)

                cost += act_in_bytes + act_out_bytes + quant_bytes + kernel_bytes
            if invalid:
                break
        if invalid:
            continue

        eff_par = _estimate_parallelism(tile_count)
        score = float(cost) / eff_par

        if (
            best_score is None
            or score < best_score
            or (
                score == best_score
                and best_tile_count is not None
                and tile_count > best_tile_count
            )
            or (
                score == best_score
                and tile_count == (best_tile_count or 0)
                and stripe_count < len(best_stripes or [])
            )
        ):
            best_score = score
            best_cost = cost
            best_stripes = candidate
            best_tile_count = tile_count

    if best_stripes is None:
        # Fallback to the minimal feasible stripes.
        best_stripes = _build_stripes(min_stripes)
        if not best_stripes:
            raise ValueError(
                f"[tiler] Cannot build any valid H stripes (H_out={h_out}, max_h_tile={hw.max_h_tile})."
            )

    return best_stripes


# ---------------------------------------------------------------------------
# Conv2D tiling
# ---------------------------------------------------------------------------


def _tile_conv2d(
    op: VcOp,
    program: VcProgram,
    layout_info: LayerLayoutInfo,
    layer_id: int,
    hw: HwConfig,
) -> tuple[List[TileDesc], LayerConfig]:
    """Tile a Conv2D/Pointwise/DW/Pool op into H + Cout stripes."""

    # Logical geometry from layout lowering
    ifm_h = layout_info.ifm_h
    ifm_w = layout_info.ifm_w
    ofm_h = layout_info.ofm_h
    ofm_w = layout_info.ofm_w
    cin = layout_info.cin
    cout = layout_info.cout
    c4_in = layout_info.c4_in
    c4_out = layout_info.c4_out

    kh, kw = op.kernel
    sh, sw = op.stride
    pt, pb, pl, pr = op.padding

    # Compatibility restriction: some historical RTL versions required even IFM_H/IFM_W for stride=2.
    # Default is relaxed; enable hw.stride2_requires_even_ifm to keep the old behavior.
    if (
        hw.stride2_requires_even_ifm
        and (sh, sw) == (2, 2)
        and op.op_type in ("conv2d", "pointwise_conv", "depthwise_conv", "dwconv")
    ):
        if (ifm_h % 2) != 0 or (ifm_w % 2) != 0:
            raise ValueError(
                f"[tiler] {op.op_type} '{op.name}': stride=2 requires even IFM_H/IFM_W "
                f"(hw.stride2_requires_even_ifm=1), got IFM_H={ifm_h}, IFM_W={ifm_w}."
            )

    # No W tiling: W_TILE == W_out
    w_tile = ofm_w

    # IBUF single-line capacity:
    # - For regular conv/pwconv, Cin is not tiled, thus Cin*IFM_W must fit into ibuf_line_bytes.
    # - For channel-wise ops (avgpool/maxpool/dwconv), allow tiling by Cout/Cin slice so each tile fits.
    max_cout_tile_by_ibuf: int | None = None
    if op.op_type in ("avg_pool", "avgpool", "max_pool", "maxpool", "depthwise_conv", "dwconv"):
        max_c = hw.ibuf_line_bytes // max(1, ifm_w)
        max_c = (max_c // 4) * 4
        if max_c < 4:
            raise ValueError(
                f"[tiler] {op.op_type} '{op.name}': ibuf_line_bytes={hw.ibuf_line_bytes} too small "
                f"for IFM_W={ifm_w} (max Cin per line={max_c})."
            )
        max_cout_tile_by_ibuf = int(max_c)
    else:
        if cin * ifm_w > hw.ibuf_line_bytes:
            raise ValueError(
                f"[tiler] {op.op_type} '{op.name}': Cin*IFM_W={cin * ifm_w} exceeds "
                f"ibuf_line_bytes={hw.ibuf_line_bytes}."
            )

    qmode = getattr(op, "qmode", None) or "INT8"

    override = _match_op_override(op)
    max_cout_tile_limit = max_cout_tile_by_ibuf
    if override and override.max_cout_tile is not None:
        if max_cout_tile_limit is None:
            max_cout_tile_limit = override.max_cout_tile
        else:
            max_cout_tile_limit = min(max_cout_tile_limit, override.max_cout_tile)
    force_divisible = override.force_divisible if override else None
    allow_wide_pw = bool(override and override.allow_wide_pw)
    ignore_max_c4_out = bool(override and override.ignore_max_c4_out)
    max_extra_stripes = override.max_extra_stripes if override else None

    # H stripes by line capacity + max_h_tile
    h_stripes = _compute_h_stripes(
        ifm_h=ifm_h,
        ifm_w=ifm_w,
        cin=cin,
        kernel_h=kh,
        kernel_w=kw,
        stride_h=sh,
        stride_w=sw,
        pad_top=pt,
        pad_bottom=pb,
        pad_left=pl,
        pad_right=pr,
        hw=hw,
        cluster_count=hw.cluster_count,
        op_type=op.op_type,
        cout=cout,
        qmode=qmode,
        max_cout_tile=max_cout_tile_limit,
        force_divisible=force_divisible,
        allow_wide_pw=allow_wide_pw,
        ignore_max_c4_out=ignore_max_c4_out,
        max_extra_stripes=max_extra_stripes,
    )

    # Cout stripes by WBUF capacity (aware of H stripe count for cluster balancing)
    cout_stripes = _compute_cout_stripes(
        op_type=op.op_type,
        cin=cin,
        cout=cout,
        ifm_h=ifm_h,
        ifm_w=ifm_w,
        kernel_h=kh,
        kernel_w=kw,
        stride_h=sh,
        stride_w=sw,
        qmode=qmode,
        hw=hw,
        cluster_count=hw.cluster_count,
        h_stripe_count=len(h_stripes),
        max_cout_tile=max_cout_tile_limit,
        force_divisible=force_divisible,
        allow_wide_pw=allow_wide_pw,
        ignore_max_c4_out=ignore_max_c4_out,
        max_extra_stripes=max_extra_stripes,
    )

    tiles: List[TileDesc] = []
    tile_index_local = 0

    prefer_param_major = op.op_type in ("conv2d", "depthwise_conv", "dwconv")

    def emit_tile(y_index: int, h_tile: int, co_start: int, co_end: int) -> None:
        nonlocal tile_index_local
        h_in_tile = h_tile * sh + kh - 1
        w_in_tile = ifm_w  # no W tiling

        # Tile-level padding (hardware semantics): only the first H stripe keeps top pad;
        # only the last H stripe keeps bottom pad. W is not tiled, so left/right are constant.
        pad_top_tile = pt if y_index == 0 else 0
        pad_bottom_tile = pb if y_index + h_tile >= ofm_h else 0
        pad_left_tile = pl
        pad_right_tile = pr

        cout_tile = co_end - co_start
        c4_out_start = co_start // 4
        c4_out_end = (co_end + 3) // 4
        # Pool ops and DWConv are channel-wise ops.
        # If Cout is tiled, the same channel slice must be selected on the input:
        #   - Cin_tile == Cout_tile
        #   - C4_IN slice matches C4_OUT slice (and FI_ADDR is offset accordingly later).
        if op.op_type in ("avg_pool", "avgpool", "max_pool", "maxpool", "depthwise_conv", "dwconv"):
            cin_tile = cout_tile
            c4_in_start = c4_out_start
            c4_in_end = c4_out_end
        else:
            cin_tile = cin
            c4_in_start = 0
            c4_in_end = c4_in

        tile = TileDesc(
            tile_id=tile_index_local,
            layer_id=layer_id,
            op_name=op.name,
            op_type=op.op_type,
            h_tile=h_tile,
            w_tile=w_tile,
            y_index=y_index,
            pad_top=pad_top_tile,
            pad_bottom=pad_bottom_tile,
            pad_left=pad_left_tile,
            pad_right=pad_right_tile,
            cin=cin_tile,
            cout=cout_tile,
            co_start=co_start,
            co_end=co_end,
            c4_in_start=c4_in_start,
            c4_in_end=c4_in_end,
            c4_out_start=c4_out_start,
            c4_out_end=c4_out_end,
            kernel_h=kh,
            kernel_w=kw,
            stride_h=sh,
            stride_w=sw,
            h_in=h_in_tile,
            w_in=w_in_tile,
            act_type=getattr(op, "activation", None),
            input_name=op.inputs[0] if op.inputs else "",
            output_name=op.outputs[0] if op.outputs else "",
            weight_name=getattr(op, "weight", "") or "",
            bias_name=getattr(op, "bias", "") or "",
            metadata={"qmode": qmode},
        )
        tiles.append(tile)
        tile_index_local += 1

    if prefer_param_major:
        for co_start, co_end in cout_stripes:
            for y_index, h_tile in h_stripes:
                emit_tile(y_index, h_tile, co_start, co_end)
    else:
        for y_index, h_tile in h_stripes:
            for co_start, co_end in cout_stripes:
                emit_tile(y_index, h_tile, co_start, co_end)

    tiles = _schedule_tiles_for_clusters(tiles, hw.cluster_count)

    # Build LayerConfig
    layer_cfg = LayerConfig(
        layer_id=layer_id,
        name=op.name,
        op_type=op.op_type,
        act_type=getattr(op, "activation", None),
        ifm_name=op.inputs[0] if op.inputs else "",
        ofm_name=op.outputs[0] if op.outputs else "",
        ifm_h=ifm_h,
        ifm_w=ifm_w,
        ofm_h=ofm_h,
        ofm_w=ofm_w,
        cin=cin,
        cout=cout,
        c4_in=c4_in,
        c4_out=c4_out,
        stride_h=sh,
        stride_w=sw,
        pad_top=pt,
        pad_bottom=pb,
        pad_left=pl,
        pad_right=pr,
        fi_stride=layout_info.fi_stride,
        fo_stride=layout_info.fo_stride,
        logical_layout=layout_info.layout,
        physical_layout="NCHWc4",
        qmode=qmode,
        quant_table=[],  # can be filled by backend using QuantTable
        kernel_h=kh,
        kernel_w=kw,
        input_name=op.inputs[0] if op.inputs else "",
        output_name=op.outputs[0] if op.outputs else "",
        weight_name=getattr(op, "weight", "") or "",
        bias_name=getattr(op, "bias", "") or "",
    )

    return tiles, layer_cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tile_program(
    program: VcProgram,
    layout_info: Dict[str, LayerLayoutInfo] | None = None,
    hw: HwConfig | None = None,
    target: str = "venuscore-v1",
) -> TilePlan:
    """
    Produce a TilePlan for the given program.
    """
    if layout_info is None:
        layout_info = {}
    if hw is None:
        hw = default_hw_config()

    tiles: List[TileDesc] = []
    tiles_by_op: Dict[str, List[TileDesc]] = {}
    layers: List[LayerConfig] = []

    layer_id = 0

    for op in program.ops:
        if isinstance(op, (VcConv2D, VcPointwiseConv, VcDepthwiseConv, VcAvgPool, VcMaxPool)):
            info = layout_info.get(op.name)
            if info is None:
                raise ValueError(
                    f"[tiler] Missing LayerLayoutInfo for Conv/Pointwise/Depthwise/Pool '{op.name}'. "
                    "Did you forget to run lower_layouts()?"
                )
            layer_tiles, layer_cfg = _tile_conv2d(op, program, info, layer_id, hw)
            tiles.extend(layer_tiles)
            tiles_by_op[op.name] = list(layer_tiles)
            layers.append(layer_cfg)
            layer_id += 1
        else:
            # Non-NPU ops are ignored here.
            continue

    _assign_tile_indices_global(tiles, tiles_by_op)

    return TilePlan(tiles=tiles, tiles_by_op=tiles_by_op, layers=layers)


def _assign_tile_indices_global(
    tiles: List[TileDesc],
    tiles_by_op: Dict[str, List[TileDesc]],
) -> None:
    """
    Assign global tile_index and per-layer first/last flags.

    Important: keep the existing tile order intact. The order of TilePlan.tiles
    directly becomes the uOP emission order (and thus affects multi-cluster
    scheduling). The tiler is responsible for producing a good order.
    """

    for idx, tile in enumerate(tiles):
        tile.tile_index = idx

    # Mark per-layer first/last in the current uOP stream order.
    layer_first: Dict[int, TileDesc] = {}
    layer_last: Dict[int, TileDesc] = {}
    for tile in tiles:
        layer_first.setdefault(tile.layer_id, tile)
        layer_last[tile.layer_id] = tile

    for t in layer_first.values():
        t.is_first_in_layer = True
    for t in layer_last.values():
        t.is_last_in_layer = True

    # Keep per-op tiles in uOP order for param planning / metadata mapping.
    for _, op_tiles in tiles_by_op.items():
        op_tiles.sort(key=lambda t: t.tile_index)
