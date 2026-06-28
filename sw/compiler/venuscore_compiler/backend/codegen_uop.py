# -*- coding: utf-8 -*-
"""
uOP generation for VenusCore backend.

Responsibilities:
  - Convert TilePlan + MemoryPlan into a list of semantic uOPs (:class:`Uop`);
  - Preserve strict alignment with VenusCore ISA semantics (H+Y version);
  - Leave binary encoding to venuscore_compiler.isa.encoder.

Notes:
  - FI_STRIDE / FO_STRIDE are taken **only** from LayerConfig and represent
    whole-layer strides (IFM_H*IFM_W, H_out*W_out). No tile-local fallback is
    used here to avoid silently diverging from the ISA spec.
  - FI_ADDR / FO_ADDR are base addresses for the IFM/OFM *sub-region* of the
    layer:
        FI_ADDR = IFM_BASE_layer + C4_IN_group * FI_STRIDE * 4
        FO_ADDR = OFM_BASE_layer + C4_OUT_group * FO_STRIDE * 4
    Currently Cin tiling is disabled and C4_IN_group is always 0, while Cout
    tiling is supported via tile.c4_out_start.

  - FIRST_FLAG / LAST_FLAG semantics:
      * FIRST_FLAG marks the first uOP in a submitted uOP stream (typically one
        NPU subgraph / one driver submit).
      * LAST_FLAG marks the last uOP in that stream.

  - SYNC semantics:
      * SYNC is asserted on the last uOP of the last tile of a layer, acting as
        a layer-level barrier (i.e. do not start issuing next-layer uOPs until
        all previous work is complete).
"""

from __future__ import annotations

from typing import List

from venuscore_compiler.backend.memory_planner import MemoryPlan
from venuscore_compiler.isa import encoder
from venuscore_compiler.isa.layout_spec import Activation, Opcode, QMode
from venuscore_compiler.isa.uop_format import Uop
from venuscore_compiler.midend.types import TileDesc, TilePlan


# Map midend op_type strings to ISA opcodes.
_OPCODE_MAP = {
    "conv2d": Opcode.CONV2D,
    "depthwise_conv": Opcode.DWCONV,
    "dwconv": Opcode.DWCONV,
    "pointwise_conv": Opcode.PWCONV,
    "avg_pool": Opcode.AVGPOOL,
    "avgpool": Opcode.AVGPOOL,
    "max_pool": Opcode.MAXPOOL,
    "maxpool": Opcode.MAXPOOL,
    "fully_connected": Opcode.MATMUL_FC,
    "matmul_fc": Opcode.MATMUL_FC,
    "matmul": Opcode.MATMUL_FC,
}


# =============================================================================
# Public API
# =============================================================================


def generate_uops(
    tile_plan: TilePlan,
    memory_plan: MemoryPlan,
    target: str = "venuscore-v1",
) -> List[Uop]:
    """
    Generate semantic uOPs from a TilePlan and MemoryPlan.

    Args:
        tile_plan:
            Midend tiling result, containing:
              - tiles: List[TileDesc]
              - layers: List[LayerConfig] (referenced by tile.layer_id)
        memory_plan:
            Backend memory allocation result, providing:
              - ifm_offsets / ofm_offsets
              - param_offsets / param_base
              - layer_configs (layer_id -> LayerConfig)
        target:
            Reserved for future target-specific tuning; currently unused.

    Returns:
        List of semantic :class:`Uop` objects ready for encoding.
    """
    _ = target  # reserved, not used yet

    uops: List[Uop] = []

    for idx, tile in enumerate(tile_plan.tiles):
        layer_cfg = memory_plan.layer_configs.get(tile.layer_id)
        if layer_cfg is None:
            raise ValueError(f"Missing LayerConfig for tile layer_id={tile.layer_id}")

        opcode = _OPCODE_MAP.get(tile.op_type, Opcode.NOP)
        act_enum = _map_activation(getattr(layer_cfg, "act_type", None))
        qmode_enum = _map_qmode(getattr(layer_cfg, "qmode", None))
        fi_stride_val, fo_stride_val = _compute_layer_strides(layer_cfg)

        # FIRST/LAST mark the boundaries of the submitted uOP stream (NPU subgraph).
        is_first_in_stream = idx == 0
        is_last_in_stream = idx == (len(tile_plan.tiles) - 1)

        # Layer barrier: only on the last uOP of the last tile of a layer.
        sync_flag = bool(tile.is_last_in_layer)

        uops.append(
            Uop(
                opcode=opcode,
                act=act_enum,
                qmode=qmode_enum,
                # Layer / tile flags
                first_flag=is_first_in_stream,
                last_flag=is_last_in_stream,
                sync=sync_flag,
                # Geometry
                h_tile=tile.h_tile,
                w_tile=tile.w_tile,
                y_index=tile.y_index,
                c4_in=int(tile.c4_in_end - tile.c4_in_start),
                # Cout is tiled, so C4_OUT is derived from tile.cout_tile.
                c4_out=(tile.cout_tile + 3) // 4,
                # Explicit IFM dims (ISA W6)
                ifm_h=int(layer_cfg.ifm_h),
                ifm_w=int(layer_cfg.ifm_w),
                # DMA precalc hints (ISA W7, word counts)
                actdma_line_words=int(layer_cfg.ifm_w) * int(tile.c4_in_end - tile.c4_in_start),
                outdma_line_words=int(tile.w_tile) * ((tile.cout_tile + 3) // 4),
                # Stride / padding
                stride_h=tile.stride_h,
                stride_w=tile.stride_w,
                pad_top=tile.pad_top,
                pad_bottom=tile.pad_bottom,
                pad_left=tile.pad_left,
                pad_right=tile.pad_right,
                # Addresses / strides
                fi_addr=_compute_fi_addr_tile(tile, memory_plan, fi_stride_val),
                fo_addr=_compute_fo_addr_tile(tile, memory_plan, fo_stride_val),
                param_addr=memory_plan.param_offsets.get(
                    (tile.op_name, tile.tile_id),
                    memory_plan.param_base,
                ),
                fi_stride=fi_stride_val,
                fo_stride=fo_stride_val,
                # Auxiliary flags (not encoded in ISA bitfields)
                flags={
                    "tile_index": tile.tile_index,
                    "first_flag": 1 if is_first_in_stream else 0,
                    "last_flag": 1 if is_last_in_stream else 0,
                    "sync": 1 if sync_flag else 0,
                    "is_first_in_layer": 1 if tile.is_first_in_layer else 0,
                    "is_last_in_layer": 1 if tile.is_last_in_layer else 0,
                },
            )
        )

    return uops


def encode_uops(uops: List[Uop]) -> bytes:
    """
    Encode a list of semantic uOPs into their binary form.

    This is a thin wrapper around venuscore_compiler.isa.encoder.encode_uops,
    provided for backward compatibility.
    """
    return encoder.encode_uops(uops)


# =============================================================================
# Helpers
# =============================================================================


def _compute_layer_strides(layer_cfg: object) -> tuple[int, int]:
    """
    Return (fi_stride, fo_stride) from LayerConfig.

    These strides are whole-layer IFM/OFM plane strides and must match the
    NCHWc4 memory layout:

        FI_STRIDE = IFM_H * IFM_W
        FO_STRIDE = H_out * W_out

    The midend is responsible for computing and filling these fields.
    """
    if not hasattr(layer_cfg, "fi_stride") or not hasattr(layer_cfg, "fo_stride"):
        raise ValueError("LayerConfig must provide fi_stride and fo_stride.")

    fi_stride = int(layer_cfg.fi_stride)
    fo_stride = int(layer_cfg.fo_stride)

    if fi_stride <= 0 or fo_stride <= 0:
        raise ValueError(f"Invalid layer strides: fi_stride={fi_stride}, fo_stride={fo_stride}")

    return fi_stride, fo_stride


def _compute_fi_addr_tile(tile: TileDesc, memory_plan: MemoryPlan, fi_stride: int) -> int:
    """
    Compute FI_ADDR for the tile.

    ISA contract (H+Y version):
        - FI_ADDR is the base address of the IFM plane (optionally offset by
          C4_IN group when Cin tiling is enabled).
        - The vertical position is carried by Y_INDEX and interpreted by
          hardware together with STRIDE/PAD/FI_STRIDE.

    Therefore FI_ADDR must remain constant across H-tiles of the same layer
    (for a fixed input tensor and C4 group), regardless of tile.y_index.
    """
    base = memory_plan.ifm_offsets.get(tile.input_name, 0)
    c4_in_start = int(getattr(tile, "c4_in_start", 0))
    return base + c4_in_start * fi_stride * 4


def _compute_fo_addr_tile(tile: TileDesc, memory_plan: MemoryPlan, fo_stride: int) -> int:
    """
    Compute FO_ADDR for the tile.

    ISA contract (H+Y version):
        - FO_ADDR is the base address of the OFM plane for the given C4_OUT
          group. The vertical position is carried by Y_INDEX and interpreted
          by hardware together with FO_STRIDE.

    Therefore FO_ADDR must remain constant across H-tiles of the same layer
    (for a fixed output tensor and C4 group), regardless of tile.y_index.
    """
    base = memory_plan.ofm_offsets.get(tile.output_name, 0)
    return base + tile.c4_out_start * fo_stride * 4


def _map_activation(act: str | None) -> Activation | None:
    """
    Map string activation in LayerConfig to Activation enum.

    Supported strings:
        - None or unknown -> Activation.NONE
        - "relu"          -> Activation.RELU
        - "relu6"         -> Activation.RELU6
    """
    if act is None:
        return Activation.NONE

    act_lower = act.lower()
    if act_lower == "relu":
        return Activation.RELU
    if act_lower == "relu6":
        return Activation.RELU6
    return Activation.NONE


def _map_qmode(qmode: str | None) -> QMode | None:
    """
    Map qmode string in LayerConfig to QMode enum.

    Supported strings:
        - None or unknown -> QMode.INT8
        - "INT8"          -> QMode.INT8
        - "INT4"          -> QMode.INT4
        - "INT2"          -> QMode.INT2
    """
    if qmode is None:
        return QMode.INT8

    qm = qmode.upper()
    if qm == "INT8":
        return QMode.INT8
    if qm == "INT4":
        return QMode.INT4
    if qm == "INT2":
        return QMode.INT2
    return QMode.INT8
