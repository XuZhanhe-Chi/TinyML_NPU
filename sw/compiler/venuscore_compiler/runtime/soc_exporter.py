# -*- coding: utf-8 -*-
"""
SoC exporters for compiled VenusCore artifacts.

Responsibilities:
  - Provide simple binary concatenation for quick SoC bring-up.
  - Provide C header array export so firmware can embed uOPs/params directly.
Dependencies:
  * Depends on: runtime.binary_format.CompiledArtifact, pathlib.Path.
  * Used by: SoC integration scripts/firmware build steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from venuscore_compiler.isa.uop_format import UOP_SIZE_BYTES
from venuscore_compiler.runtime.binary_format import CompiledArtifact

__all__ = ["export_to_soc_binary", "export_c_header_arrays"]


def export_to_soc_binary(artifact: CompiledArtifact, out_path: str | Path) -> None:
    """Export compiled binaries suitable for a RISC-V SoC runtime."""

    # TODO: Define binary packaging per SoC ABI.
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Concatenate uOP and parameter payloads; real format will likely prepend metadata.
    path.write_bytes(artifact.uop_binary + artifact.param_block)
    # Also emit a companion C header for firmware embedding by default.
    header_path = path.with_suffix(path.suffix + ".h")
    export_c_header_arrays(artifact, header_path)


def _bytes_to_u32_words(data: bytes) -> List[int]:
    """
    Convert a bytes payload into little-endian uint32 words, padding with zeros.

    This is convenient for SoC firmware that prefers word-addressed loads.
    """
    if not data:
        return []
    padded = data
    if (len(padded) % 4) != 0:
        padded = padded + b"\x00" * (4 - (len(padded) % 4))
    words: List[int] = []
    for i in range(0, len(padded), 4):
        words.append(int.from_bytes(padded[i : i + 4], byteorder="little", signed=False))
    return words


def _format_c_u32_array(name: str, data: bytes) -> List[str]:
    """Format a bytes payload as a C static const uint32_t array (little-endian, padded)."""

    words = _bytes_to_u32_words(data)
    return _format_c_u32_array_words(name, words)


def _format_c_u32_array_words(name: str, words: List[int], decl_attrs: str | None = None) -> List[str]:
    """Format a uint32 word list as a C static const uint32_t array."""

    attrs = f" {decl_attrs}" if decl_attrs else ""
    lines: List[str] = [f"static const uint32_t {name}[]{attrs} = {{"]
    if words:
        for i in range(0, len(words), 8):
            chunk = words[i : i + 8]
            body = ", ".join(f"0x{w:08X}u" for w in chunk)
            lines.append(f"    {body},")
    lines.append("};")
    return lines


def export_c_header_arrays(
    artifact: CompiledArtifact, out_path: str | Path, header_guard: str = "VENUSCORE_COMPILER_BUNDLE_H"
) -> None:
    """
    Emit a C header that embeds uOP and param payloads as uint32_t word arrays.

    The header contains:
      - static const uint32_t uops_words[] (little-endian u32 words)
      - static const uint32_t params_words[] (little-endian u32 words)
      - size/offset macros derived from artifact.metadata and payload lengths
    """

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Precompute word views once to avoid repeated conversions for large blobs.
    uops_words = _bytes_to_u32_words(artifact.uop_binary)
    params_words = _bytes_to_u32_words(artifact.param_block)

    def _format_header_comment(meta: dict) -> List[str]:
        uop_count = int(meta.get("uop_count", 0))
        if not uop_count and artifact.uop_binary:
            uop_count = len(artifact.uop_binary) // UOP_SIZE_BYTES
        if not uop_count and artifact.uops:
            uop_count = len(artifact.uops)

        layer_quant = meta.get("layer_quant", [])
        layer_count = len(layer_quant) if isinstance(layer_quant, list) else 0

        address_mode = str(meta.get("address_mode", "absolute"))

        plan = meta.get("plan") if isinstance(meta, dict) else None
        has_plan = isinstance(plan, dict)
        plan_tensor_count = 0
        plan_step_count = 0
        plan_npu_step_count = 0
        plan_cpu_step_count = 0
        plan_alias_step_count = 0
        plan_arena_bytes = 0
        if has_plan:
            try:
                plan_tensor_count = int(len(plan.get("tensors", []) or []))
                steps = list(plan.get("steps", []) or [])
                plan_step_count = int(len(steps))
                for s in steps:
                    if not isinstance(s, dict):
                        continue
                    st = str(s.get("type", "NPU"))
                    if st == "NPU":
                        plan_npu_step_count += 1
                    elif st == "CPU":
                        plan_cpu_step_count += 1
                    elif st == "ALIAS":
                        plan_alias_step_count += 1
                plan_arena_bytes = int(plan.get("arena_bytes", 0))
            except Exception:
                # Best-effort only; never fail header generation due to comment formatting.
                plan_tensor_count = 0
                plan_step_count = 0
                plan_npu_step_count = 0
                plan_cpu_step_count = 0
                plan_alias_step_count = 0
                plan_arena_bytes = 0

        lines: List[str] = [
            "/*",
            " * VenusCore Compiler 自动生成文件（bundle.h）",
            " *",
            " * 本头文件用于固件/驱动侧直接嵌入编译产物：",
            " *   - uops_words[]  ：32B 定长 uOP 指令流（小端，8×u32 / uOP）",
            " *   - params_words[]：Param Block blob（小端 u32 words，末尾零填充到 4B 对齐）",
            " *",
            f" * Target              : {meta.get('target', 'unknown')}",
            f" * Address mode        : {address_mode} ({'需要重定位' if address_mode == 'offset' else '绝对地址'})",
            f" * uOP count           : {uop_count}",
            f" * uops_words          : {len(uops_words)} words ({len(artifact.uop_binary)} bytes)",
            f" * params_words        : {len(params_words)} words ({len(artifact.param_block)} bytes)",
            f" * activation_peak     : {meta.get('activation_peak_bytes', 0)} bytes",
            f" * input               : base={meta.get('input_base', 0)} size={meta.get('input_size', 0)}",
            f" * output              : base={meta.get('output_base', 0)} size={meta.get('output_size', 0)}",
            f" * param_base          : {meta.get('param_base', 0)}",
            f" * Layers (quant)      : {layer_count}",
            f" * Plan                : {'enabled' if has_plan else 'disabled'}",
            f" *   - tensors         : {plan_tensor_count}",
            f" *   - steps           : {plan_step_count} (NPU={plan_npu_step_count}, CPU={plan_cpu_step_count}, ALIAS={plan_alias_step_count})",
            f" *   - arena_bytes     : {plan_arena_bytes}",
            " *",
            " * 若 ADDRESS_MODE_OFFSET==1，固件需对 uOP 做地址重定位（relocation）：",
            " *   - word[3] (W3) PARAM_ADDR += param_base",
            " *   - word[4] (W4) FI_ADDR    += act_base",
            " *   - word[5] (W5) FO_ADDR    += act_base",
            " *",
            " * 注意：params_words[]/uops_words[] 在 SoC/XIP 场景下通常需要至少 64B 对齐（并建议放入专用 section），",
            " * 否则某些硬件/集成路径下可能因 Param Block 对齐假设导致结果错误。",
            " */",
            "",
        ]

        if isinstance(layer_quant, list) and layer_quant:
            lines.append("/* 层摘要（来自 metadata.layer_quant）：")
            for i, layer in enumerate(layer_quant):
                if not isinstance(layer, dict):
                    continue
                name = str(layer.get("name", ""))
                op_type = str(layer.get("op_type", ""))
                qmode = layer.get("qmode", None)
                qt = layer.get("quant_table", [])
                qt_len = len(qt) if isinstance(qt, list) else 0
                preview = ""
                if isinstance(qt, list) and qt:
                    first = qt[0]
                    if isinstance(first, (list, tuple)) and len(first) == 3:
                        preview = f" first=(bias={first[0]}, scale_u16={first[1]}, shift={first[2]})"
                lines.append(f" *   L{i:02d}: op_type={op_type} qmode={qmode} qlen={qt_len}{preview} name={name}")
            lines.append(" */")
            lines.append("")

        tile_uop_map = meta.get("tile_uop_map", [])
        if isinstance(tile_uop_map, list) and tile_uop_map:
            # Summarize per-op tiling counts (avoid printing every tile entry for large graphs).
            by_op: dict[str, int] = {}
            for ent in tile_uop_map:
                if not isinstance(ent, dict):
                    continue
                op = str(ent.get("op", ""))
                if not op:
                    continue
                by_op[op] = by_op.get(op, 0) + 1
            lines.append("/* Tile 摘要（来自 metadata.tile_uop_map）：")
            for op, cnt in sorted(by_op.items(), key=lambda kv: kv[0]):
                lines.append(f" *   {op}: tiles={cnt}")
            lines.append(" */")
            lines.append("")

        return lines

    # Prepare macros.
    meta = artifact.metadata if isinstance(artifact.metadata, dict) else {}

    # This file is often checked into firmware repos; provide a short human-readable header.
    lines: List[str] = _format_header_comment(meta)

    def _get_meta(key: str) -> int:
        val = meta.get(key, 0)
        try:
            return int(val)
        except Exception:
            return 0

    address_mode = str(meta.get("address_mode", "absolute"))
    address_mode_offset = 1 if address_mode == "offset" else 0

    plan = meta.get("plan")
    has_plan = isinstance(plan, dict)
    tensor_count = 0
    step_count = 0
    arena_bytes = 0
    uops_len_words = 0
    params_len_words = 0
    quant_scale_count = 0
    if has_plan:
        try:
            tensor_count = int(len(plan.get("tensors", [])))
            step_count = int(len(plan.get("steps", [])))
            arena_bytes = int(plan.get("arena_bytes", 0))
            uops_len_words = int(plan.get("uops_len_words", 0))
            params_len_words = int(plan.get("params_len_words", 0))
            quant_scale_count = int(len(plan.get("quant_scales", [])))
        except Exception:
            tensor_count = 0
            step_count = 0
            arena_bytes = 0
            uops_len_words = 0
            params_len_words = 0
            quant_scale_count = 0

    # Header guard and includes first, then macros (for readability).
    lines += [
        f"#ifndef {header_guard}",
        f"#define {header_guard}",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "/* ---------------- 宏定义（由编译产物导出） ---------------- */",
        f"#define ADDRESS_MODE_OFFSET {address_mode_offset}u",
        f"#define UOPS_LEN_BYTES {len(artifact.uop_binary)}u",
        f"#define PARAMS_LEN_BYTES {len(artifact.param_block)}u",
        f"#define UOPS_LEN_WORDS {len(uops_words)}u",
        f"#define PARAMS_LEN_WORDS {len(params_words)}u",
        f"#define ACTIVATION_PEAK_BYTES {_get_meta('activation_peak_bytes')}u",
        f"#define WEIGHT_BYTES {_get_meta('weight_bytes')}u",
        f"#define INPUT_BASE {_get_meta('input_base')}u",
        f"#define INPUT_SIZE {_get_meta('input_size')}u",
        f"#define OUTPUT_BASE {_get_meta('output_base')}u",
        f"#define OUTPUT_SIZE {_get_meta('output_size')}u",
        f"#define PARAM_BASE {_get_meta('param_base')}u",
        f"#define PARAM_BLOCK_SIZE {len(artifact.param_block)}u",
        f"#define UOP_BINARY_SIZE {len(artifact.uop_binary)}u",
        f"#define VC_HAS_PLAN {(1 if has_plan else 0)}u",
        "",
    ]

    if has_plan:
        lines += [
            f"#define VC_PLAN_TENSOR_COUNT {tensor_count}u",
            f"#define VC_PLAN_STEP_COUNT {step_count}u",
            f"#define VC_PLAN_ARENA_BYTES {arena_bytes}u",
            f"#define VC_PLAN_UOPS_LEN_WORDS {uops_len_words}u",
            f"#define VC_PLAN_PARAMS_LEN_WORDS {params_len_words}u",
            f"#define VC_PLAN_QUANT_SCALE_COUNT {quant_scale_count}u",
            "",
        ]

    if has_plan:
        lines.append("/* ---------------- typedef（Plan ABI，v1） ---------------- */")
        # Minimal stable ABI for plan execution (v1). Keep in sync with venus_driver.h.
        lines += [
            "typedef enum { VC_STEP_NPU = 0, VC_STEP_CPU = 1, VC_STEP_ALIAS = 2 } vc_step_type_t;",
            "typedef enum { VC_CPU_ADD = 0, VC_CPU_CONCAT_C = 1 } vc_cpu_kernel_t;",
            "typedef enum { VC_ACT_NONE = 0, VC_ACT_RELU = 1, VC_ACT_RELU6 = 2 } vc_cpu_activation_t;",
            "#define VC_STEP_DESC_HAS_CPU_ACTIVATION 1",
            "",
            "typedef struct {",
            "    uint16_t tensor_id;",
            "    uint16_t quant_index; /* 0xFFFF if unused */",
            "    uint32_t offset_bytes;",
            "    uint32_t size_bytes;",
            "    uint16_t n, c, h, w; /* logical NCHW */",
            "} vc_tensor_desc_t;",
            "",
            "typedef struct {",
            "    uint8_t step_type;   /* vc_step_type_t */",
            "    uint8_t cpu_kernel;  /* vc_cpu_kernel_t (only if step_type==CPU) */",
            "    uint8_t cpu_activation; /* vc_cpu_activation_t */",
            "    uint8_t axis;        /* concat axis (v1 uses C==1) */",
            "    uint8_t input_count;",
            "    uint8_t output_count;",
            "    uint16_t inputs[4];",
            "    uint16_t outputs[2];",
            "    uint32_t uop_off_words;",
            "    uint32_t uop_words;",
            "    uint32_t param_off_words;",
            "    uint32_t param_words;",
            "} vc_step_desc_t;",
            "",
            "typedef struct {",
            "    const vc_tensor_desc_t* tensors;",
            "    uint32_t tensor_count;",
            "    const vc_step_desc_t* steps;",
            "    uint32_t step_count;",
            "    uint32_t arena_bytes;",
            "    uint32_t uops_len_words;",
            "    uint32_t params_len_words;",
            "    const float* quant_scales; /* optional, may be NULL */",
            "    uint32_t quant_scale_count;",
            "} vc_plan_t;",
            "",
        ]

    # Keep small const scalars near the top; put large blobs at the end.
    lines.append("/* ---------------- const 常量（小） ---------------- */")
    lines += [
        f"static const size_t uops_words_len_words = {len(uops_words)}u;",
        f"static const size_t params_words_len_words = {len(params_words)}u;",
        "",
    ]

    if has_plan:
        lines.append("/* ---------------- const 常量（Plan） ---------------- */")
        def _u16(x: object, default: int = 0xFFFF) -> int:
            try:
                v = int(x)  # type: ignore[arg-type]
                if v < 0:
                    return default
                return v & 0xFFFF
            except Exception:
                return default

        def _u32(x: object, default: int = 0) -> int:
            try:
                v = int(x)  # type: ignore[arg-type]
                if v < 0:
                    return default
                return v & 0xFFFF_FFFF
            except Exception:
                return default

        tensors = plan.get("tensors", [])
        steps = plan.get("steps", [])

        lines.append("static const vc_tensor_desc_t VC_TENSORS[] = {")
        if isinstance(tensors, list):
            for t in tensors:
                if not isinstance(t, dict):
                    continue
                shape = t.get("shape", [1, 1, 1, 1])
                if not (isinstance(shape, list) and len(shape) == 4):
                    shape = [1, 1, 1, 1]
                qidx = t.get("quant_index", None)
                lines.append(
                    "    {"
                    f"{_u16(t.get('tensor_id', 0), 0)}, "
                    f"{_u16(0xFFFF if qidx is None else qidx, 0xFFFF)}, "
                    f"{_u32(t.get('offset_bytes', 0))}u, "
                    f"{_u32(t.get('size_bytes', 0))}u, "
                    f"{_u16(shape[0], 1)}, {_u16(shape[1], 1)}, {_u16(shape[2], 1)}, {_u16(shape[3], 1)}"
                    "},"
                )
        lines.append("};")
        lines.append("")

        def _step_type(s: dict) -> int:
            st = str(s.get("type", "NPU"))
            if st == "NPU":
                return 0
            if st == "CPU":
                return 1
            if st == "ALIAS":
                return 2
            return 0

        def _cpu_kernel(s: dict) -> int:
            k = str(s.get("kernel", "ADD"))
            if k == "ADD":
                return 0
            if k == "CONCAT_C":
                return 1
            return 0

        def _cpu_activation(s: dict) -> int:
            a = str(s.get("activation", "NONE")).upper()
            if a == "NONE":
                return 0
            if a == "RELU":
                return 1
            if a == "RELU6":
                return 2
            return 0

        def _pad_list(vals: list[int], n: int, pad: int) -> list[int]:
            out = list(vals)[:n]
            while len(out) < n:
                out.append(pad)
            return out

        lines.append("static const vc_step_desc_t VC_STEPS[] = {")
        if isinstance(steps, list):
            for s in steps:
                if not isinstance(s, dict):
                    continue
                inputs = _pad_list([_u16(x, 0xFFFF) for x in (s.get("inputs", []) or [])], 4, 0xFFFF)
                outputs = _pad_list([_u16(x, 0xFFFF) for x in (s.get("outputs", []) or [])], 2, 0xFFFF)
                st = _step_type(s)
                ck = _cpu_kernel(s) if st == 1 else 0
                ca = _cpu_activation(s) if st == 1 else 0
                axis = _u32(s.get("axis", 1)) & 0xFF
                uop_off_words = _u32(s.get("uop_off_words", 0))
                uop_words = _u32(s.get("uop_words", 0))
                param_off_words = _u32(s.get("param_off_words", 0))
                param_words = _u32(s.get("param_words", 0))
                lines.append(
                    "    {"
                    f"{st}u, {ck}u, {ca}u, {axis}u, "
                    f"{_u32(len([x for x in inputs if x != 0xFFFF]))}u, "
                    f"{_u32(len([x for x in outputs if x != 0xFFFF]))}u, "
                    f"{{{', '.join(f'{v}u' for v in inputs)}}}, "
                    f"{{{', '.join(f'{v}u' for v in outputs)}}}, "
                    f"{uop_off_words}u, {uop_words}u, {param_off_words}u, {param_words}u"
                    "},"
                )
        lines.append("};")
        lines.append("")

        qscales = plan.get("quant_scales", [])
        lines.append("static const float VC_QUANT_SCALES[] = {")
        if isinstance(qscales, list) and qscales:
            for s in qscales:
                try:
                    fs = float(s)
                except Exception:
                    fs = 0.0
                # Use a stable string form; avoid scientific notation when possible.
                fs_str = f"{fs:.10g}"
                if ("." not in fs_str) and ("e" not in fs_str.lower()):
                    fs_str += ".0"
                lines.append(f"    {fs_str}f,")
        lines.append("};")
        lines.append("")

        lines += [
            "static const vc_plan_t VC_PLAN = {",
            "    VC_TENSORS,",
            "    VC_PLAN_TENSOR_COUNT,",
            "    VC_STEPS,",
            "    VC_PLAN_STEP_COUNT,",
            "    VC_PLAN_ARENA_BYTES,",
            "    VC_PLAN_UOPS_LEN_WORDS,",
            "    VC_PLAN_PARAMS_LEN_WORDS,",
            "    VC_QUANT_SCALES,",
            "    VC_PLAN_QUANT_SCALE_COUNT,",
            "};",
            "",
        ]

    # Keep large binary blobs at the end of the header to make it easier to
    # navigate in IDEs and reduce diff noise when the ABI changes.
    lines.append("/* ---------------- const 常量（大：二进制 blob） ---------------- */")
    blob_align = "__attribute__((section(\".venus_uops\"), aligned(64)))"
    lines += _format_c_u32_array_words("uops_words", uops_words, decl_attrs=blob_align)
    lines.append("")
    blob_align = "__attribute__((section(\".venus_params\"), aligned(64)))"
    lines += _format_c_u32_array_words("params_words", params_words, decl_attrs=blob_align)
    lines.append("")

    lines.append("")
    lines.append(f"#endif /* {header_guard} */")

    path.write_text("\n".join(lines), encoding="utf-8")
