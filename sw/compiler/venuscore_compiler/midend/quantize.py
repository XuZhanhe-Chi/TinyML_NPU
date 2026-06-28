# -*- coding: utf-8 -*-
"""
Midend quantization lowering: map logical tensor scales to hardware-friendly (mul, shift).

According to the VenusCore NPU software architecture, the midend is responsible for:

  * Reading logical quantization info from VcTensor (symmetric int8 only);
  * For each NPU-accelerated layer (Conv / DWConv / PWConv / FC), computing
    per-output-channel quantization coefficients:

        bias_s32, scale_u16, shift_u6

    such that the effective real scale is approximated by:

        real_scale  ~=  scale_u16 / 2**shift_u6

    and

        real_scale  =  (ifm_scale * weight_scale_oc) / ofm_scale

    where:
      - ifm_scale:   per-tensor activation scale of the input tensor;
      - weight_scale_oc: per-channel (or per-tensor) scale of the weight tensor;
      - ofm_scale:   per-tensor activation scale of the output tensor.

  * Emitting:
      - QuantTable: op_name -> list[(bias_s32, scale_u16, shift_u6)]
      - QModeMap:   op_name -> "INT8" / "INT4" / "INT2" (currently only "INT8").

The backend Param Block builder will then directly serialize these tuples into
the Param Block layout expected by the hardware.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from venuscore_compiler.ir.ops import (
    VcAvgPool,
    VcConv2D,
    VcDepthwiseConv,
    VcPointwiseConv,
    VcFullyConnected,
    VcMaxPool,
)
from venuscore_compiler.ir.program import VcProgram

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

# op_name -> list[(bias_s32, scale_u16, shift_u6)]
QuantTable = Dict[str, List[Tuple[int, int, int]]]

# op_name -> qmode string ("INT8"/"INT4"/"INT2"...)
QModeMap = Dict[str, str]

__all__ = ["QuantTable", "QModeMap", "compute_quant_params"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tensor(program: VcProgram, name: str):
    try:
        return program.tensors[name]
    except KeyError as e:
        raise ValueError(f"[quantize] Unknown tensor name '{name}' referenced in IR.") from e


def _get_per_tensor_scale(tensor) -> float:
    """
    Fetch per-tensor symmetric scale from VcTensor.

    For VenusCore we assume:
      - symmetric_per_tensor for activations (IFM/OFM),
      - symmetric_per_tensor or symmetric_per_channel for weights.

    This helper enforces per-tensor and raises if tensor is not quantized properly.
    """
    if not getattr(tensor, "is_quantized", None) or not tensor.is_quantized():
        return 1.0

    if hasattr(tensor, "is_per_tensor_quant") and tensor.is_per_tensor_quant():
        scale = tensor.scale
        # scale may be a scalar or 1-element sequence; normalize to float
        if isinstance(scale, (list, tuple)):
            if len(scale) != 1:
                return 1.0
            try:
                return float(scale[0])
            except Exception:
                return 1.0
        try:
            return float(scale)
        except Exception:
            return 1.0

    # Per-channel activations are not supported (for now).
    return 1.0


def _get_weight_scale_for_oc(weight_tensor, oc: int) -> float:
    """
    Get per-output-channel weight scale.

    If weight tensor is per-channel quantized, we assume q_axis maps to Cout
    (typically axis 0 under NCHW/OIHW conventions). Otherwise we fall back to
    per-tensor scale.
    """
    if not getattr(weight_tensor, "is_quantized", None) or not weight_tensor.is_quantized():
        return 1.0

    if hasattr(weight_tensor, "is_per_channel_quant") and weight_tensor.is_per_channel_quant():
        # Use tensor helper if available
        if hasattr(weight_tensor, "get_scale_for_channel"):
            return float(weight_tensor.get_scale_for_channel(oc))

        # Fallback: direct indexing into .scale
        scale_seq = weight_tensor.scale
        try:
            return float(scale_seq[oc])
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"[quantize] Failed to fetch per-channel scale for tensor '{weight_tensor.name}', oc={oc}."
            ) from exc

    # Per-tensor weight quantization
    return _get_per_tensor_scale(weight_tensor)


def _extract_bias_int32(bias_tensor, oc: int) -> int:
    """
    Extract bias value for given output channel as int32.

    We expect bias tensor to be either:
      - per-channel: shape like [1, Cout, 1, 1] or [Cout], or
      - scalar broadcast: then we reuse the same value for all oc.
    """
    if bias_tensor is None:
        return 0

    data = getattr(bias_tensor, "data", None)
    if data is None:
        return 0

    # Try a few common layouts
    try:
        # [1, Cout, 1, 1]
        return int(data[0][oc][0][0])
    except Exception:
        pass

    try:
        # [Cout]
        return int(data[oc])
    except Exception:
        pass

    try:
        # scalar
        return int(data)
    except Exception:
        return 0


def _decompose_scale(real_scale: float) -> Tuple[int, int]:
    """
    Decompose a positive real scale into (scale_u16, shift_u6) such that:

        real_scale ~= scale_u16 / 2**shift_u6

    with:
        1 <= scale_u16 <= 0xFFFF
        0 <= shift_u6  <= 63

    We perform a small search over shift to minimize relative error, with sensible
    clamping for extreme values.
    """
    if real_scale <= 0.0:
        raise ValueError(f"[quantize] real_scale must be positive, got {real_scale}.")

    # Try all shift values from 0 to 63, keep the best approximation.
    best_scale = None
    best_shift = 0
    best_err = float("inf")

    for shift in range(64):
        scale = real_scale * (1 << shift)
        if scale < 1.0 or scale > 0xFFFF:
            continue
        scale_int = int(round(scale))
        if scale_int < 1 or scale_int > 0xFFFF:
            continue
        approx = scale_int / float(1 << shift)
        err = abs(approx - real_scale) / real_scale
        if err < best_err:
            best_err = err
            best_scale = scale_int
            best_shift = shift
            # Early exit for near-perfect match
            if err < 1e-6:
                break

    if best_scale is None:
        # If we fail to find an in-range pair (very extreme real_scale),
        # we clamp to the nearest representable.
        if real_scale >= 1.0:
            best_scale = 0xFFFF
            best_shift = 0
        else:
            # For very small real_scale, push it to the smallest positive value.
            best_scale = 1
            best_shift = 63

    return best_scale, best_shift


def _check_int32(value: int, name: str) -> None:
    _check_range(value, -2**31, 2**31 - 1, name)


def _check_range(value: int, lo: int, hi: int, name: str) -> None:
    if value < lo or value > hi:
        raise ValueError(f"[quantize] {name}={value} out of range [{lo}, {hi}]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_quant_params(program: VcProgram) -> Tuple[QuantTable, QModeMap]:
    """
    Compute per-op, per-output-channel quantization parameters.

    For each supported op (Conv / DepthwiseConv / PointwiseConv / FullyConnected),
    this function computes:

        bias_s32, scale_u16, shift_u6

    and stores them in QuantTable[op.name][oc].

    It also decides the layer-level QMODE tag (currently always "INT8") and stores
    it in QModeMap[op.name].
    """
    quant_table: QuantTable = {}
    qmode_map: QModeMap = {}

    for op in program.ops:
        # Handle NPU-mapped ops. Weight-bearing ops use (ifm_scale * w_scale) / ofm_scale,
        # while Pool uses (ifm_scale / ofm_scale) because there are no weights.
        if isinstance(op, (VcConv2D, VcDepthwiseConv, VcPointwiseConv, VcFullyConnected)):
            if not op.weight:
                raise ValueError(f"[quantize] Op '{op.name}' has no weight tensor assigned.")

            # Resolve tensors
            ifm_tensor = _get_tensor(program, op.inputs[0])
            ofm_tensor = _get_tensor(program, op.outputs[0])
            weight_tensor = _get_tensor(program, op.weight)
            bias_tensor = _get_tensor(program, op.bias) if op.bias else None

            # Logical geometry: get Cout from OFM tensor
            # We interpret shape/layout via OFM tensor helpers if needed, but since
            # normalize already guarantees 4D with N==1, we can take C directly.
            ofm_shape = getattr(ofm_tensor, "shape")
            if len(ofm_shape) != 4:
                raise ValueError(
                    f"[quantize] OFM tensor '{ofm_tensor.name}' for op '{op.name}' must be 4D, got {ofm_shape}."
                )
            cout = ofm_shape[1]  # assume NCHW after normalize

            # Activation scales (per-tensor)
            ifm_scale = _get_per_tensor_scale(ifm_tensor)
            ofm_scale = _get_per_tensor_scale(ofm_tensor)

            # Per-OC coefficients
            layer_coeffs: List[Tuple[int, int, int]] = []

            for oc in range(cout):
                w_scale = _get_weight_scale_for_oc(weight_tensor, oc)
                real_scale = (ifm_scale * w_scale) / ofm_scale

                scale_u16, shift_u6 = _decompose_scale(real_scale)

                bias_s32 = _extract_bias_int32(bias_tensor, oc)
                _check_int32(bias_s32, f"bias[{oc}]")

                layer_coeffs.append((bias_s32, scale_u16, shift_u6))

            quant_table[op.name] = layer_coeffs
            # Preserve any pre-annotated qmode override (for example, pool
            # rounding markers carried by example compile flows); otherwise
            # default to INT8.
            qmode_map[op.name] = str(getattr(op, "qmode", None) or "INT8").upper()
        elif isinstance(op, (VcAvgPool, VcMaxPool)):
            # Pool ops have no weights. The hardware applies pooling in integer domain and then
            # uses the same SFU model: out_i8 ~= (acc_i32 * scale_u16) >> shift_u6.
            #
            # We therefore approximate:
            #   out_scale_real = ifm_scale / ofm_scale
            # and use bias=0 for all output channels.
            if not op.inputs or not op.outputs:
                raise ValueError(f"[quantize] Pool '{op.name}' missing inputs or outputs.")

            ifm_tensor = _get_tensor(program, op.inputs[0])
            ofm_tensor = _get_tensor(program, op.outputs[0])

            ofm_shape = getattr(ofm_tensor, "shape")
            if len(ofm_shape) != 4:
                raise ValueError(
                    f"[quantize] OFM tensor '{ofm_tensor.name}' for op '{op.name}' must be 4D, got {ofm_shape}."
                )
            cout = int(ofm_shape[1])  # assume NCHW after normalize

            ifm_scale = _get_per_tensor_scale(ifm_tensor)
            ofm_scale = _get_per_tensor_scale(ofm_tensor)

            real_scale = ifm_scale / ofm_scale
            scale_u16, shift_u6 = _decompose_scale(real_scale)
            layer_coeffs = [(0, scale_u16, shift_u6) for _ in range(cout)]

            quant_table[op.name] = layer_coeffs
            qmode_map[op.name] = str(getattr(op, "qmode", None) or "INT8").upper()

    return quant_table, qmode_map
