# -*- coding: utf-8 -*-
"""
Build binary Param Blocks for VenusCore tiles (Quant Coeff + aligned weights).

Depends on:
  - backend.memory_planner.MemoryPlan for per-tile PARAM_ADDR offsets
  - midend.tiler.TilePlan / TileDesc / LayerConfig for geometry/channel ranges
  - param_data (weight/bias dict) provided by caller
  - QuantTable for per-OC quantization parameters

Layout follows VenusCore ISA and memory map specification:

    PARAM_ADDR ->
        [ Quant Coeff Block (Cout_tile * 8B) ]
        [ Alignment Padding to 16B ]
        [ Weight Data Block (layout depends on op_type) ]

    - CONV2D / PWCONV:
        Outer:  co in [co_start, co_end)
        Inner:  c4_in, kh, kw
        Each word = 4× int8 weights along Cin dimension.

    - DWCONV:
        Cin_tile = Cin_dw (not tiled, Cin_dw = layer_cfg.cin)
        C4_dw    = Cin_dw / 4
        Outer:   g in [0, C4_dw), ch_in_group in [0,3], kw in [0,2]
        Word = {top, mid, bot, 0} for that channel and kw.

    - MATMUL / FC:
        TODO: currently not implemented; code will raise NotImplementedError
              to avoid silently wrong layouts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from venuscore_compiler.backend.memory_planner import MemoryPlan
from venuscore_compiler.midend.types import TilePlan, TileDesc, LayerConfig
from venuscore_compiler.midend.quantize import QuantTable

WORD_BYTES = 4
ALIGNMENT = 16


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def build_param_block(
    tile_plan: TilePlan,
    memory_plan: MemoryPlan,
    param_data: Dict[str, Dict[str, Any]],
    quant_table: QuantTable | None = None,
    target: str = "venuscore-v1",
) -> bytes:
    """
    Build the full Param Block blob (params.bin) for a given TilePlan.

    This function will:
      - Iterate tiles in tile_plan.tiles_by_op order;
      - For each tile:
        * Align current cursor to MemoryPlan.param_offsets[(op_name, tile_id)]
          if present, otherwise use current cursor as PARAM_ADDR;
        * Build per-tile Param Block (Quant + Weight) according to op_type;
      - Update memory_plan.param_size to the final blob size.

    Args:
        tile_plan:
            Tiling plan containing TileDesc and LayerConfig list.
        memory_plan:
            MemoryPlan with param_base and per-tile param_offsets pre-assigned.
        param_data:
            Dict[name, tensor_dict], where tensor_dict at least provides:
              - "shape": List[int]
              - "data":  flat List[int] (int8 / int32, etc.)
        quant_table:
            Global quant table (LayerConfig.quant_table is preferred if present).
        target:
            Reserved for future targets; currently unused.

    Returns:
        params.bin content as bytes.
    """
    _ = target  # Currently unused, kept for future extension.

    buf = bytearray()
    # Fast lookup: layer_id -> LayerConfig
    layer_cfg_map = {cfg.layer_id: cfg for cfg in getattr(tile_plan, "layers", [])}
    emitted_keys: Dict[Tuple[str, int, int], int] = {}

    for op_name, tiles in tile_plan.tiles_by_op.items():
        for tile in tiles:
            share_key = (op_name, tile.co_start, tile.co_end)
            if share_key in emitted_keys:
                # This tile reuses an already-emitted weight/quant slice; skip emission.
                continue

            # Expected PARAM_ADDR for this tile (absolute address)
            key = (op_name, tile.tile_id)
            expected_offset = memory_plan.param_offsets.get(
                key,
                memory_plan.param_base + len(buf),
            )
            current_offset = memory_plan.param_base + len(buf)

            if current_offset < expected_offset:
                # Pad gap with zeros
                buf.extend(b"\x00" * (expected_offset - current_offset))
            elif current_offset > expected_offset:
                raise ValueError(
                    f"Param block offset mismatch for op {op_name!r}, tile {tile.tile_id}: "
                    f"current={current_offset}, expected={expected_offset}"
                )

            layer_cfg = layer_cfg_map.get(tile.layer_id)
            block = _build_param_block_for_tile(tile, layer_cfg, param_data, quant_table)
            buf.extend(block)
            emitted_keys[share_key] = expected_offset

    memory_plan.param_size = len(buf)
    return bytes(buf)


# -----------------------------------------------------------------------------
# Per-tile Param Block construction
# -----------------------------------------------------------------------------


def _build_param_block_for_tile(
    tile: TileDesc,
    layer_cfg: LayerConfig | None,
    param_data: Dict[str, Dict[str, Any]],
    quant_table: QuantTable | None,
) -> bytes:
    """Emit Param Block bytes for a single tile.

    Layout:
        [ Quant Coeff Block (Cout_tile * 8B) ]
        [ 16B alignment padding ]
        [ Weight Data Block (op_type-specific) ]
    """
    weight_tensor = param_data.get(tile.weight_name) if getattr(tile, "weight_name", None) else None
    bias_tensor = param_data.get(tile.bias_name) if getattr(tile, "bias_name", None) else None

    # Channel range for this tile in logical Cout indices.
    co_start = tile.co_start
    co_end = tile.co_end
    cout_tile = co_end - co_start
    if cout_tile <= 0:
        raise ValueError(f"Invalid Cout tile range for tile {tile.op_name!r}: [{co_start}, {co_end})")

    block = bytearray()

    # -------------------------------------------------------------------------
    # 1) Quant Coeff Block: Cout_tile × 8B
    #    Layout per OC (tile-local index oc = 0..Cout_tile-1):
    #      8B = [Bias_s32 (low 32-bit)] + [Scale_u16 + Shift_u6 + Reserved_10 (high 32-bit)]
    # -------------------------------------------------------------------------
    for oc in range(cout_tile):
        co_global = co_start + oc
        bias_val = _get_bias(bias_tensor, co_global)
        scale_u16, shift_u6 = _get_quant_coeff(quant_table, layer_cfg, co_global)

        upper32 = ((0 & 0x3FF) << (16 + 6)) | ((shift_u6 & 0x3F) << 16) | (scale_u16 & 0xFFFF)
        value64 = (upper32 << 32) | (bias_val & 0xFFFFFFFF)
        block.extend(value64.to_bytes(8, byteorder="little", signed=False))

    # -------------------------------------------------------------------------
    # 2) Align to 16B for WeightBase
    # -------------------------------------------------------------------------
    weight_base = (len(block) + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
    if weight_base > len(block):
        block.extend(b"\x00" * (weight_base - len(block)))

    # -------------------------------------------------------------------------
    # 3) Weight Data Block: op_type-specific layout
    # -------------------------------------------------------------------------
    op_type = tile.op_type.lower()

    if op_type in ("avg_pool", "avgpool", "max_pool", "maxpool"):
        # Pool ops have no weight block. Keep only the Quant Coeff Block (and padding).
        return bytes(block)

    if op_type in ("conv2d", "pointwise_conv"):
        if weight_tensor is None:
            raise ValueError(f"Weight tensor {tile.weight_name!r} not found for tile {tile.op_name!r}.")
        _emit_weights_conv2d(block, weight_tensor, tile, layer_cfg)
    elif op_type in ("depthwise_conv", "dwconv"):
        if weight_tensor is None:
            raise ValueError(f"Weight tensor {tile.weight_name!r} not found for tile {tile.op_name!r}.")
        _emit_weights_dwconv(block, weight_tensor, tile, layer_cfg)
    elif op_type in ("fully_connected", "matmul", "matmul_fc"):
        # To avoid silent wrong outputs, FC/MATMUL is not implemented yet and must raise.
        raise NotImplementedError(
            f"Param Block weight layout for op_type={tile.op_type!r} "
            "is not implemented yet. Please implement MATMUL/FC layout per spec."
        )
    else:
        raise ValueError(f"Unsupported op_type for Param Block: {tile.op_type!r}")

    return bytes(block)


# -----------------------------------------------------------------------------
# Weight emitters (per op type)
# -----------------------------------------------------------------------------


def _emit_weights_conv2d(
    block: bytearray,
    weight_tensor: Dict[str, Any],
    tile: TileDesc,
    layer_cfg: LayerConfig | None,
) -> None:
    """
    Emit Weight Data Block for CONV2D / PWCONV.

    Assumes logical weight tensor layout: [Cout, Cin, Kh, Kw] (NCHW).
    Cin is not tiled (Cin_tile = layer_cfg.cin), Cout is sliced by [co_start, co_end).
    For each tile:
        Outer loops: co_global, c4_in, kh, kw
        Each word packs 4 int8 weights along Cin dimension.
    """
    shape = _get_shape(weight_tensor)
    data = _get_data(weight_tensor)

    if shape is None or data is None:
        raise ValueError("Weight tensor is missing 'shape' or 'data' field.")

    if len(shape) != 4:
        raise ValueError(f"CONV2D weight tensor expects 4D shape [Cout, Cin, Kh, Kw], got {shape!r}")

    co_total, cin_total, kh, kw = shape

    # Cin is not tiled; if layer_cfg provided, sanity check.
    if layer_cfg is not None and cin_total != layer_cfg.cin:
        raise ValueError(
            f"CONV2D weight Cin mismatch: weight Cin={cin_total}, layer_cfg.cin={layer_cfg.cin}"
        )

    co_start = tile.co_start
    co_end = tile.co_end
    cin_tile = cin_total
    c4_in_tile = (cin_tile + 3) // 4 if cin_tile > 0 else 0

    for co in range(co_start, co_end):
        if not (0 <= co < co_total):
            raise ValueError(f"co index {co} out of range for weight tensor Cout={co_total}")
        for c4 in range(c4_in_tile):
            for y in range(kh):
                for x in range(kw):
                    # Gather 4 weights along Cin
                    w_bytes: List[int] = []
                    for k in range(4):
                        cin_idx = c4 * 4 + k
                        if cin_idx < cin_total:
                            w_val = _get_weight_4d(data, shape, co, cin_idx, y, x)
                        else:
                            w_val = 0
                        w_bytes.append(w_val & 0xFF)

                    word = (
                        (w_bytes[0] & 0xFF)
                        | ((w_bytes[1] & 0xFF) << 8)
                        | ((w_bytes[2] & 0xFF) << 16)
                        | ((w_bytes[3] & 0xFF) << 24)
                    )
                    block.extend(word.to_bytes(WORD_BYTES, byteorder="little", signed=False))


def _emit_weights_dwconv(
    block: bytearray,
    weight_tensor: Dict[str, Any],
    tile: TileDesc,
    layer_cfg: LayerConfig | None,
) -> None:
    """
    Emit Weight Data Block for DWCONV (Depthwise 3×3).

    Logical weight tensor layout assumed: [Cin_dw, Kh=3, Kw=3].

    Layout per spec:

        Cin_dw = Cin_tile (channel-wise slicing, aligned to C4)
        C4_dw  = Cin_dw / 4   (Cin_dw must be multiple of 4)

        for g in [0 .. C4_dw-1]:
          for ch_in_group in [0..3]:
            ch = 4*g + ch_in_group
            for kw in [0..2]:
                word_W_dw(ch, kw) = {
                    K_dw[ch][0][kw],  // top
                    K_dw[ch][1][kw],  // mid
                    K_dw[ch][2][kw],  // bot
                    0                 // padding
                }

        word_index = (g * 4 + ch_in_group) * 3 + kw
        byte_addr  = WeightBase + word_index * WORD_BYTES
    """
    shape = _get_shape(weight_tensor)
    data = _get_data(weight_tensor)

    if shape is None or data is None:
        raise ValueError("DWCONV weight tensor is missing 'shape' or 'data' field.")
    if len(shape) == 4 and shape[1] == 1:
        # Common ONNX layout [Cin, 1, Kh, Kw]; squeeze the singleton.
        cin_dw, _, kh, kw = shape
        data = np.array(data).reshape(cin_dw, kh, kw).tolist()
        shape = (cin_dw, kh, kw)
    if len(shape) != 3:
        raise ValueError(f"DWCONV weight tensor expects 3D shape [Cin_dw, Kh, Kw], got {shape!r}")

    cin_total, kh, kw = shape
    if kh != 3 or kw != 3:
        raise ValueError(f"DWCONV expects 3x3 kernel, got Kh={kh}, Kw={kw}")

    if layer_cfg is not None and cin_total != layer_cfg.cin:
        raise ValueError(f"DWCONV weight Cin mismatch: weight Cin={cin_total}, layer_cfg.cin={layer_cfg.cin}")

    cin_dw = int(getattr(tile, "cout", 0))
    if cin_dw <= 0:
        raise ValueError(f"DWCONV tile has invalid Cin_dw={cin_dw}")
    if cin_dw % 4 != 0:
        raise ValueError(f"DWCONV requires Cin_dw multiple of 4, got Cin_dw={cin_dw}")

    co_start = int(getattr(tile, "co_start", 0))
    co_end = int(getattr(tile, "co_end", 0))
    if co_end - co_start != cin_dw:
        raise ValueError(f"DWCONV tile channel slice mismatch: (co_end-co_start)={co_end-co_start} Cin_dw={cin_dw}")
    if co_start < 0 or co_end > int(cin_total):
        raise ValueError(f"DWCONV tile channel slice out of range: [{co_start},{co_end}) with Cin_total={cin_total}")

    c4_dw = cin_dw // 4

    for g in range(c4_dw):
        for ch_in_group in range(4):
            ch_global = co_start + 4 * g + ch_in_group
            if not (0 <= ch_global < cin_total):
                raise ValueError(f"DWCONV channel index {ch_global} out of range (Cin_total={cin_total})")
            for kw_idx in range(kw):  # 0..2
                top = _get_weight_3d(data, shape, ch_global, 0, kw_idx)
                mid = _get_weight_3d(data, shape, ch_global, 1, kw_idx)
                bot = _get_weight_3d(data, shape, ch_global, 2, kw_idx)
                w_bytes = [
                    top & 0xFF,
                    mid & 0xFF,
                    bot & 0xFF,
                    0,
                ]
                word = (
                    (w_bytes[0] & 0xFF)
                    | ((w_bytes[1] & 0xFF) << 8)
                    | ((w_bytes[2] & 0xFF) << 16)
                    | ((w_bytes[3] & 0xFF) << 24)
                )
                block.extend(word.to_bytes(WORD_BYTES, byteorder="little", signed=False))


# -----------------------------------------------------------------------------
# Quant / bias helpers
# -----------------------------------------------------------------------------


def _get_bias(bias_tensor: Dict[str, Any] | None, co_global: int) -> int:
    """Fetch bias coefficient for a given global Cout index."""
    if bias_tensor is None:
        return 0

    data = _get_data(bias_tensor)
    shape = _get_shape(bias_tensor)

    if data is None or shape is None:
        return 0

    # Common case: bias stored as [1, Cout, 1, 1]
    if len(shape) == 4:
        n, c, h, w = (int(x) for x in shape)
        # Try nested indexing first (e.g. data[1][Cout][1][1]).
        try:
            return int(data[0][co_global][0][0])
        except Exception:
            # FIX: Some frontends store bias as a flat list but still carry a 4D shape
            # like [1, Cout, 1, 1]. In that case, interpret the payload as flattened.
            if isinstance(data, list) and h == 1 and w == 1:
                if n == 1 and len(data) == c and 0 <= co_global < c:
                    return int(data[co_global])
                if c == 1 and len(data) == n and 0 <= co_global < n:
                    return int(data[co_global])

    if len(shape) == 1:
        # shape: [Cout]
        if not (0 <= co_global < shape[0]):
            return 0
        return int(data[co_global])

    # Fallback: treat first dimension as Cout, ignore remaining dims.
    if not (0 <= co_global < shape[0]):
        return 0
    # Flatten along remaining dims.
    inner = 1
    for d in shape[1:]:
        inner *= int(d)
    base = co_global * inner
    # Assume bias is uniform across inner dims, just take first.
    try:
        return int(data[base])
    except Exception:
        return 0


def _get_quant_coeff(
    quant_table: QuantTable | None,
    layer_cfg: LayerConfig | None,
    co_global: int,
) -> Tuple[int, int]:
    """
    Get (scale_u16, shift_u6) for given Cout index.

    Priority:
        1) layer_cfg.quant_table if present
        2) global quant_table if present
        3) fallback to (scale=1, shift=0)
    """
    # 1) LayerConfig.quant_table
    if layer_cfg is not None and getattr(layer_cfg, "quant_table", None) is not None:
        qt = layer_cfg.quant_table
        if 0 <= co_global < len(qt):
            bias_s32, scale_u16, shift_u6 = qt[co_global]
            _ = bias_s32  # bias handled separately
            return int(scale_u16) & 0xFFFF, int(shift_u6) & 0x3F

    # 2) Global QuantTable (assumed List[Tuple[bias, scale, shift]] too)
    if quant_table is not None:
        if 0 <= co_global < len(quant_table):
            bias_s32, scale_u16, shift_u6 = quant_table[co_global]
            _ = bias_s32
            return int(scale_u16) & 0xFFFF, int(shift_u6) & 0x3F

    # 3) Fallback
    return 1, 0


# -----------------------------------------------------------------------------
# Generic tensor helpers
# -----------------------------------------------------------------------------


def _get_shape(tensor_like: Dict[str, Any] | None):
    """Fetch shape from a tensor-like dict."""
    if tensor_like is None:
        return None
    return tensor_like.get("shape")


def _get_data(tensor_like: Dict[str, Any] | None):
    """Fetch data field from a tensor-like dict."""
    if tensor_like is None:
        return None
    return tensor_like.get("data")


def _get_weight_4d(
    data: List[int],
    shape: List[int],
    co: int,
    cin: int,
    kh: int,
    kw: int,
) -> int:
    """Index weight from a [Cout, Cin, Kh, Kw] tensor; data may be nested."""
    cout, cin_total, kh_total, kw_total = shape
    if not (0 <= co < cout and 0 <= cin < cin_total and 0 <= kh < kh_total and 0 <= kw < kw_total):
        return 0
    try:
        return int(data[co][cin][kh][kw])
    except Exception:
        # Fallback if data is flattened
        try:
            idx = (((co * cin_total) + cin) * kh_total + kh) * kw_total + kw
            return int(data[idx])
        except Exception:
            return 0


def _get_weight_3d(
    data: List[int],
    shape: List[int],
    ch: int,
    kh: int,
    kw: int,
) -> int:
    """Index weight from a [Cin, Kh, Kw] tensor (flat data)."""
    cin_total, kh_total, kw_total = shape
    if not (0 <= ch < cin_total):
        raise IndexError(f"ch={ch} out of range [0,{cin_total})")
    if not (0 <= kh < kh_total):
        raise IndexError(f"kh={kh} out of range [0,{kh_total})")
    if not (0 <= kw < kw_total):
        raise IndexError(f"kw={kw} out of range [0,{kw_total})")

    try:
        return int(data[ch][kh][kw])
    except Exception:
        idx = (ch * kh_total + kh) * kw_total + kw
        return int(data[idx])
