# -*- coding: utf-8 -*-
"""
Parse a generated bundle.h and provide host-side utilities.

This is intended for host validation and unit tests:
- Parse macros such as ADDRESS_MODE_OFFSET / UOPS_LEN_WORDS / PARAMS_LEN_WORDS
- Parse uops_words[] / params_words[] arrays
- Relocate uOP address words (W3/W4/W5) for offset addressing mode
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import struct
from pathlib import Path
from typing import Dict, List


_RE_DEFINE = re.compile(
    r"^\s*#define[ \t]+([A-Za-z_][A-Za-z0-9_]*)(?:[ \t]+(.*?))?[ \t]*$",
    re.MULTILINE,
)


def _parse_int_literal(value: str) -> int:
    s = value.strip()
    # Drop common unsigned suffixes: 123u, 123ul, 0x10U, etc.
    s = re.sub(r"(?i)(u|ul|ull|l|ll)\b", "", s).strip()
    # Allow parentheses in simple macro bodies.
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return int(s, 0)


def parse_defines(text: str) -> Dict[str, str]:
    """
    Parse simple C preprocessor defines from bundle.h.

    Important: use horizontal whitespace only so a macro without a value does not
    accidentally consume the next line (CRLF contains newlines which are also
    matched by \\s).
    """
    out: Dict[str, str] = {}
    for m in _RE_DEFINE.finditer(text):
        key = m.group(1)
        val = m.group(2) if m.group(2) is not None else ""
        out[key] = val.strip()
    return out


_RE_ARRAY_START = re.compile(
    r"static\s+const\s+uint32_t\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*\]\s*(?:__attribute__\s*\(\(.*?\)\)\s*)?=\s*\{",
    re.MULTILINE,
)


def _extract_uint32_array(text: str, name: str) -> List[int]:
    m = re.search(
        rf"static\s+const\s+uint32_t\s+{re.escape(name)}\s*\[\s*\]\s*(?:__attribute__\s*\(\(.*?\)\)\s*)?=\s*\{{",
        text,
        flags=re.MULTILINE,
    )
    if not m:
        raise ValueError(f"Array not found: {name}")
    start = m.end()
    end = text.find("};", start)
    if end < 0:
        raise ValueError(f"Array terminator not found for: {name}")
    body = text[start:end]
    # Accept 0x..., decimal numbers, and optional separators/suffixes.
    nums = re.findall(r"0x[0-9A-Fa-f_]+|\d+", body)
    out: List[int] = []
    for tok in nums:
        out.append(int(tok.replace("_", ""), 0) & 0xFFFF_FFFF)
    return out


def words_to_uops_bytes(words: List[int]) -> bytes:
    if len(words) % 8 != 0:
        raise ValueError(f"uops_words length must be a multiple of 8, got {len(words)}")
    b = bytearray()
    for w in words:
        b.extend(struct.pack("<I", w & 0xFFFF_FFFF))
    return bytes(b)


def relocate_uops_words(words: List[int], act_base: int, param_base: int) -> List[int]:
    """
    Apply the same relocation logic as the offset-mode driver:
      - W3 (PARAM_ADDR) += param_base
      - W4 (FI_ADDR)    += act_base
      - W5 (FO_ADDR)    += act_base
    """
    if len(words) % 8 != 0:
        raise ValueError(f"uops_words length must be a multiple of 8, got {len(words)}")
    out = list(words)
    for i in range(0, len(out), 8):
        out[i + 3] = (out[i + 3] + int(param_base)) & 0xFFFF_FFFF
        out[i + 4] = (out[i + 4] + int(act_base)) & 0xFFFF_FFFF
        out[i + 5] = (out[i + 5] + int(act_base)) & 0xFFFF_FFFF
    return out


@dataclass(frozen=True)
class BundleH:
    path: Path
    defines_raw: Dict[str, str]
    uops_words: List[int]
    params_words: List[int]

    def define_int(self, key: str) -> int:
        if key not in self.defines_raw:
            raise KeyError(f"Missing define: {key}")
        return _parse_int_literal(self.defines_raw[key])


@dataclass(frozen=True)
class VcTensorDesc:
    tensor_id: int
    quant_index: int
    offset_bytes: int
    size_bytes: int
    n: int
    c: int
    h: int
    w: int


@dataclass(frozen=True)
class VcStepDesc:
    step_type: int
    cpu_kernel: int
    cpu_activation: int
    axis: int
    input_count: int
    output_count: int
    inputs: List[int]
    outputs: List[int]
    uop_off_words: int
    uop_words: int
    param_off_words: int
    param_words: int


def _extract_braced_initializer(text: str, anchor: str) -> str:
    """
    Extract the initializer body between the first '{' after anchor and the matching '};'.
    """
    idx = text.find(anchor)
    if idx < 0:
        raise ValueError(f"Anchor not found: {anchor}")
    brace = text.find("{", idx)
    if brace < 0:
        raise ValueError(f"Opening brace not found after anchor: {anchor}")

    depth = 0
    end = -1
    for i in range(brace, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise ValueError(f"Initializer braces not closed for anchor: {anchor}")

    # Expect a trailing ';' (allow whitespace/newlines).
    tail = text[end : end + 4]
    if "};" not in tail:
        # Best-effort: find the nearest '};' after end.
        semi = text.find("};", end)
        if semi < 0:
            raise ValueError(f"Initializer terminator '}};' not found for anchor: {anchor}")
        end = semi + 1  # point at the closing brace of '};'
    return text[brace + 1 : end]


def _split_top_level_entries(body: str) -> List[str]:
    """
    Split a braced initializer body into top-level "{...}" entry strings.
    """
    entries: List[str] = []
    depth = 0
    start = None
    for i, ch in enumerate(body):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                entries.append(body[start : i + 1])
                start = None
    return entries


def _extract_int_tokens(s: str) -> List[int]:
    toks = re.findall(r"0x[0-9A-Fa-f_]+|\d+", s)
    return [int(t.replace("_", ""), 0) for t in toks]


def parse_plan_tensors(text: str) -> List[VcTensorDesc]:
    body = _extract_braced_initializer(text, "static const vc_tensor_desc_t VC_TENSORS[]")
    entries = _split_top_level_entries(body)
    out: List[VcTensorDesc] = []
    for e in entries:
        nums = _extract_int_tokens(e)
        if len(nums) != 8:
            raise ValueError(f"Unexpected VC_TENSORS entry field count: got {len(nums)} in {e!r}")
        out.append(
            VcTensorDesc(
                tensor_id=nums[0],
                quant_index=nums[1],
                offset_bytes=nums[2],
                size_bytes=nums[3],
                n=nums[4],
                c=nums[5],
                h=nums[6],
                w=nums[7],
            )
        )
    return out


def parse_plan_steps(text: str) -> List[VcStepDesc]:
    body = _extract_braced_initializer(text, "static const vc_step_desc_t VC_STEPS[]")
    entries = _split_top_level_entries(body)
    out: List[VcStepDesc] = []
    for e in entries:
        nums = _extract_int_tokens(e)
        # step_type,cpu_kernel,cpu_activation,axis,input_count,output_count,inputs[4],outputs[2],uop_off,uop_words,param_off,param_words
        if len(nums) != 16:
            raise ValueError(f"Unexpected VC_STEPS entry field count: got {len(nums)} in {e!r}")
        out.append(
            VcStepDesc(
                step_type=nums[0],
                cpu_kernel=nums[1],
                cpu_activation=nums[2],
                axis=nums[3],
                input_count=nums[4],
                output_count=nums[5],
                inputs=nums[6:10],
                outputs=nums[10:12],
                uop_off_words=nums[12],
                uop_words=nums[13],
                param_off_words=nums[14],
                param_words=nums[15],
            )
        )
    return out


def load_bundle_h(path: Path) -> BundleH:
    text = path.read_text(encoding="utf-8")
    defines_raw = parse_defines(text)
    uops_words = _extract_uint32_array(text, "uops_words")
    params_words = _extract_uint32_array(text, "params_words")
    return BundleH(path=path, defines_raw=defines_raw, uops_words=uops_words, params_words=params_words)
