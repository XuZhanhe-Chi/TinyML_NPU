#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


INPUT_SCALE = 0.127225593
OUTPUT_SCALE = 0.0589496195
INPUT_N = 1
INPUT_C = 1
INPUT_H = 50
INPUT_W = 40
OUTPUT_C = 12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantize_i8(x: np.ndarray, scale: float) -> np.ndarray:
    q = np.rint(x.astype(np.float32) / float(scale)).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)


def _expected_i8(logits: np.ndarray, scale: float) -> np.ndarray:
    y = logits.astype(np.float32)
    rounded = np.rint(y)
    if np.allclose(y, rounded, atol=1.0e-6) and float(y.min()) >= -128.0 and float(y.max()) <= 127.0:
        q = rounded.astype(np.int32)
    else:
        q = np.rint(y / float(scale)).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)


def _u8_hex_rows(values: np.ndarray, *, row_len: int = 16) -> list[str]:
    flat = values.astype(np.uint8).reshape(-1)
    rows: list[str] = []
    for off in range(0, flat.size, row_len):
        rows.append("    " + ", ".join(f"0x{int(v):02X}u" for v in flat[off : off + row_len]))
    return rows


def _u32_rows(values: list[int], *, row_len: int = 12) -> list[str]:
    rows: list[str] = []
    for off in range(0, len(values), row_len):
        rows.append("    " + ", ".join(f"{int(v)}u" for v in values[off : off + row_len]))
    return rows


def _str_rows(values: list[str]) -> list[str]:
    return ["    " + ", ".join(json.dumps(v) for v in values)]


def _write_array_2d(out: list[str], c_type: str, name: str, dims: str, values: np.ndarray) -> None:
    out.append(f"static const {c_type} {name}{dims} = {{")
    for si, sample in enumerate(values):
        out.append(f"  /* sample {si:03d} */")
        out.append("  {")
        rows = _u8_hex_rows(sample, row_len=16)
        for idx, row in enumerate(rows):
            suffix = "," if idx != len(rows) - 1 else ""
            out.append(row + suffix)
        out.append("  }" + ("," if si != values.shape[0] - 1 else ""))
    out.append("};")
    out.append("")


def generate_header(input_npy: Path, expected_npy: Path, meta_json: Path, output: Path) -> None:
    x = np.load(input_npy).astype(np.float32)
    logits = np.load(expected_npy).astype(np.float32)
    meta = json.loads(meta_json.read_text(encoding="utf-8"))

    if x.ndim != 4 or tuple(x.shape[1:]) != (INPUT_C, INPUT_H, INPUT_W):
        raise ValueError(f"input must have shape [N,{INPUT_C},{INPUT_H},{INPUT_W}], got {x.shape}")
    if logits.ndim != 2 or logits.shape[1] != OUTPUT_C:
        raise ValueError(f"expected logits must have shape [N,{OUTPUT_C}], got {logits.shape}")
    if x.shape[0] != logits.shape[0]:
        raise ValueError(f"input/logits sample count mismatch: {x.shape[0]} vs {logits.shape[0]}")

    samples = meta.get("samples")
    if not isinstance(samples, list) or len(samples) != int(x.shape[0]):
        raise ValueError("meta samples must be a list matching the NPY sample count")

    labels = np.asarray([int(s["label"]) for s in samples], dtype=np.uint8)
    source_indices = [int(s.get("sample_idx", i)) for i, s in enumerate(samples)]
    expected_i8 = _expected_i8(logits, OUTPUT_SCALE)
    expected_top1 = np.argmax(expected_i8.astype(np.int16), axis=1).astype(np.uint8)
    ref_label_correct = int(np.sum(expected_top1 == labels))
    label_hist = [int(np.sum(labels == i)) for i in range(OUTPUT_C)]
    ref_top1_hist = [int(np.sum(expected_top1 == i)) for i in range(OUTPUT_C)]

    q_input = _quantize_i8(x, INPUT_SCALE)
    compact_input = q_input[:, 0, :, :].reshape((x.shape[0], INPUT_H * INPUT_W)).astype(np.uint8)
    expected_bytes = expected_i8.astype(np.uint8)
    commands = meta.get("commands")
    if not isinstance(commands, list) or len(commands) != OUTPUT_C:
        commands = [f"class_{i}" for i in range(OUTPUT_C)]

    out: list[str] = []
    out.append("#ifndef VENUSCORE_KWS_TESTVECTOR_FPGA_H")
    out.append("#define VENUSCORE_KWS_TESTVECTOR_FPGA_H")
    out.append("")
    out.append("#include <stdint.h>")
    out.append("")
    out.append("/*")
    out.append(" * Auto-generated multi-sample FPGA testvector header.")
    out.append(" *")
    out.append(" * This file intentionally stores only quantized KWS features and reference")
    out.append(" * logits needed by the public ZYBO7010 board demo. It does not include the")
    out.append(" * training set, ONNX model, or training code.")
    out.append(" *")
    out.append(f" * input_f32.npy  sha256={_sha256(input_npy)}")
    out.append(f" * qdq_logits.npy sha256={_sha256(expected_npy)}")
    out.append(f" * meta.json      sha256={_sha256(meta_json)}")
    out.append(" */")
    out.append("")
    out.append("#define VC_KWS_SAMPLE_INDEX 0u")
    out.append(f"#define VC_KWS_SAMPLE_COUNT {int(x.shape[0])}u")
    out.append("#define VC_KWS_MIN_SAMPLE_COUNT 100u")
    out.append(f"#define VC_KWS_CLASS_COUNT {OUTPUT_C}u")
    out.append(f"#define VC_KWS_EXPECTED_LABEL_CORRECT {ref_label_correct}u")
    out.append(f"#define VC_KWS_EXPECTED_REF_TOP1_MATCH {int(x.shape[0])}u")
    out.append("")
    out.append(f"#define VC_KWS_INPUT_N {INPUT_N}u")
    out.append(f"#define VC_KWS_INPUT_C {INPUT_C}u")
    out.append(f"#define VC_KWS_INPUT_H {INPUT_H}u")
    out.append(f"#define VC_KWS_INPUT_W {INPUT_W}u")
    out.append("#define VC_KWS_INPUT_C4 1u")
    out.append("#define VC_KWS_INPUT_WORDS (VC_KWS_INPUT_C4 * VC_KWS_INPUT_H * VC_KWS_INPUT_W)")
    out.append("#define VC_KWS_INPUT_BYTES (VC_KWS_INPUT_WORDS * 4u)")
    out.append("#define VC_KWS_INPUT_COMPACT_BYTES (VC_KWS_INPUT_H * VC_KWS_INPUT_W)")
    out.append("")
    out.append("#define VC_KWS_OUTPUT_N 1u")
    out.append(f"#define VC_KWS_OUTPUT_C {OUTPUT_C}u")
    out.append("#define VC_KWS_OUTPUT_H 1u")
    out.append("#define VC_KWS_OUTPUT_W 1u")
    out.append("#define VC_KWS_OUTPUT_C4 3u")
    out.append("#define VC_KWS_OUTPUT_WORDS (VC_KWS_OUTPUT_C4 * VC_KWS_OUTPUT_H * VC_KWS_OUTPUT_W)")
    out.append("#define VC_KWS_OUTPUT_BYTES (VC_KWS_OUTPUT_WORDS * 4u)")
    out.append("")
    out.append(f"static const float VC_KWS_INPUT_SCALE = {INPUT_SCALE:.9g}f;")
    out.append(f"static const float VC_KWS_OUTPUT_SCALE = {OUTPUT_SCALE:.9g}f;")
    out.append("")
    out.append("static const uint32_t VC_KWS_META_LABEL = 0u;")
    out.append(f"static const uint32_t VC_KWS_EXPECTED_TOP1_F32 = {int(expected_top1[0])}u;")
    out.append(f"static const uint32_t VC_KWS_EXPECTED_TOP1_I8  = {int(expected_top1[0])}u;")
    out.append("")
    out.append("static const char *const VC_KWS_CLASS_NAMES[VC_KWS_CLASS_COUNT] = {")
    for row in _str_rows([str(c) for c in commands]):
        out.append(row)
    out.append("};")
    out.append("")
    out.append("static const uint8_t VC_KWS_LABELS[VC_KWS_SAMPLE_COUNT] = {")
    for row in _u8_hex_rows(labels, row_len=24):
        out.append(row + ",")
    out.append("};")
    out.append("")
    out.append("static const uint8_t VC_KWS_EXPECTED_TOP1[VC_KWS_SAMPLE_COUNT] = {")
    for row in _u8_hex_rows(expected_top1, row_len=24):
        out.append(row + ",")
    out.append("};")
    out.append("")
    out.append("static const uint32_t VC_KWS_SOURCE_SAMPLE_INDEX[VC_KWS_SAMPLE_COUNT] = {")
    for row in _u32_rows(source_indices, row_len=12):
        out.append(row + ",")
    out.append("};")
    out.append("")
    out.append("static const uint32_t VC_KWS_LABEL_HISTOGRAM[VC_KWS_CLASS_COUNT] = {")
    for row in _u32_rows(label_hist, row_len=12):
        out.append(row + ",")
    out.append("};")
    out.append("")
    out.append("static const uint32_t VC_KWS_REF_TOP1_HISTOGRAM[VC_KWS_CLASS_COUNT] = {")
    for row in _u32_rows(ref_top1_hist, row_len=12):
        out.append(row + ",")
    out.append("};")
    out.append("")
    out.append("/* Compact int8 input: [sample][H*W]. Firmware expands each byte into NCHWc4 lanes. */")
    _write_array_2d(
        out,
        "uint8_t",
        "VC_KWS_INPUT_I8",
        "[VC_KWS_SAMPLE_COUNT][VC_KWS_INPUT_COMPACT_BYTES]",
        compact_input,
    )
    out.append("/* Reference output bytes: [sample][12] signed INT8 logits stored as uint8_t. */")
    _write_array_2d(
        out,
        "uint8_t",
        "VC_KWS_EXPECTED_OUTPUT_I8",
        "[VC_KWS_SAMPLE_COUNT][VC_KWS_OUTPUT_C]",
        expected_bytes,
    )
    out.append("#endif")
    out.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact multi-sample KWS FPGA header.")
    parser.add_argument("--input-npy", type=Path, required=True)
    parser.add_argument("--expected-npy", type=Path, required=True)
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_header(args.input_npy, args.expected_npy, args.meta_json, args.output)


if __name__ == "__main__":
    main()
