# -*- coding: utf-8 -*-
"""
Binary serialization for the mixed-execution Plan (v1).

This is an optional companion output to bundle.h/metadata.json. The binary
format mirrors the C structs emitted into bundle.h by the runtime exporter.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List


_MAGIC = b"VCP2"  # VenusCore Plan (v2: adds quant_scales table)


def encode_plan_v1(plan: Dict[str, Any]) -> bytes:
    """
    Encode a plan dict (as produced by Plan.to_dict()) into a v1 binary blob.

    Layout (little-endian), version 2:
      header:
        char[4]  magic = "VCP2"
        u16      version = 2
        u16      reserved = 0
        u32      tensor_count
        u32      step_count
        u32      arena_bytes
        u32      uops_len_words
        u32      params_len_words
        u32      quant_scale_count
      quant_scales[quant_scale_count]:
        f32 per-tensor symmetric scale
      tensors[tensor_count]:
        u16 tensor_id
        u16 quant_index (0xFFFF if unused)
        u32 offset_bytes
        u32 size_bytes
        u16 n, c, h, w
      steps[step_count] (matches vc_step_desc_t packing, sizeof==36):
        u8  step_type
        u8  cpu_kernel
        u8  cpu_activation
        u8  axis
        u8  input_count
        u8  output_count
        u16 inputs[4]
        u16 outputs[2]
        u16 pad1
        u32 uop_off_words
        u32 uop_words
        u32 param_off_words
        u32 param_words
    """

    tensors = plan.get("tensors", [])
    steps = plan.get("steps", [])
    if not isinstance(tensors, list) or not isinstance(steps, list):
        raise ValueError("plan['tensors'] and plan['steps'] must be lists")

    tensor_count = len(tensors)
    step_count = len(steps)

    arena_bytes = int(plan.get("arena_bytes", 0))
    uops_len_words = int(plan.get("uops_len_words", 0))
    params_len_words = int(plan.get("params_len_words", 0))

    out = bytearray()
    quant_scales = plan.get("quant_scales", []) or []
    if not isinstance(quant_scales, list):
        raise ValueError("plan['quant_scales'] must be a list when present")

    out += _MAGIC
    out += struct.pack(
        "<HHIIIIII",
        2,
        0,
        tensor_count,
        step_count,
        arena_bytes,
        uops_len_words,
        params_len_words,
        len(quant_scales),
    )

    for s in quant_scales:
        out += struct.pack("<f", float(s))

    def _u16(x: Any, default: int = 0xFFFF) -> int:
        try:
            v = int(x)
        except Exception:
            return default
        if v < 0:
            return default
        return v & 0xFFFF

    def _u32(x: Any, default: int = 0) -> int:
        try:
            v = int(x)
        except Exception:
            return default
        if v < 0:
            return default
        return v & 0xFFFF_FFFF

    # Tensors
    for t in tensors:
        if not isinstance(t, dict):
            raise ValueError("tensor entry must be a dict")
        shape = t.get("shape", [1, 1, 1, 1])
        if not (isinstance(shape, list) and len(shape) == 4):
            raise ValueError(f"tensor shape must be 4D list, got {shape!r}")
        qidx = t.get("quant_index", None)
        out += struct.pack(
            "<HHIIHHHH",
            _u16(t.get("tensor_id", 0), 0),
            _u16(0xFFFF if qidx is None else qidx, 0xFFFF),
            _u32(t.get("offset_bytes", 0)),
            _u32(t.get("size_bytes", 0)),
            _u16(shape[0], 1),
            _u16(shape[1], 1),
            _u16(shape[2], 1),
            _u16(shape[3], 1),
        )

    def _step_type(s: dict) -> int:
        st = str(s.get("type", "NPU"))
        if st == "NPU":
            return 0
        if st == "CPU":
            return 1
        if st == "ALIAS":
            return 2
        raise ValueError(f"Unknown step type {st!r}")

    def _cpu_kernel(s: dict) -> int:
        k = str(s.get("kernel", "ADD"))
        if k == "ADD":
            return 0
        if k == "CONCAT_C":
            return 1
        raise ValueError(f"Unknown CPU kernel {k!r}")

    def _cpu_activation(s: dict) -> int:
        a = str(s.get("activation", "NONE")).upper()
        if a == "NONE":
            return 0
        if a == "RELU":
            return 1
        if a == "RELU6":
            return 2
        raise ValueError(f"Unknown CPU activation {a!r}")

    # Steps
    for s in steps:
        if not isinstance(s, dict):
            raise ValueError("step entry must be a dict")
        inputs = s.get("inputs", []) or []
        outputs = s.get("outputs", []) or []
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise ValueError("step inputs/outputs must be lists")
        if len(inputs) > 4 or len(outputs) > 2:
            raise ValueError(f"step IO exceeds v1 limits: inputs={len(inputs)} outputs={len(outputs)}")

        st = _step_type(s)
        ck = _cpu_kernel(s) if st == 1 else 0
        ca = _cpu_activation(s) if st == 1 else 0
        axis = _u16(s.get("axis", 1), 1) & 0xFF

        in_pad = [_u16(x, 0xFFFF) for x in inputs] + [0xFFFF] * (4 - len(inputs))
        out_pad = [_u16(x, 0xFFFF) for x in outputs] + [0xFFFF] * (2 - len(outputs))

        out += struct.pack(
            "<6B",
            st & 0xFF,
            ck & 0xFF,
            ca & 0xFF,
            axis & 0xFF,
            len(inputs) & 0xFF,
            len(outputs) & 0xFF,
        )
        out += struct.pack("<4H2H", *in_pad, *out_pad)
        out += struct.pack("<H", 0)  # pad1 to 4-byte alignment
        out += struct.pack(
            "<4I",
            _u32(s.get("uop_off_words", 0)),
            _u32(s.get("uop_words", 0)),
            _u32(s.get("param_off_words", 0)),
            _u32(s.get("param_words", 0)),
        )

    return bytes(out)
