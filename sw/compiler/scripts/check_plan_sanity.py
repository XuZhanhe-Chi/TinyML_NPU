# -*- coding: utf-8 -*-
"""
Host-side Plan sanity checks for bundle.h.

This script focuses on integration correctness for board/FPGA bring-up:
- Plan tensor bounds, sizes (NCHWc4 int8), and alignment
- Step bounds for uops/params segments and basic invariants
- (Optional) uOP address range checks, using either offset mode or absolute mode
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from venuscore_compiler.isa.decoder import decode_uops
from venuscore_compiler.runtime.bundle_h_parser import (
    BundleH,
    VcStepDesc,
    VcTensorDesc,
    load_bundle_h,
    parse_defines,
    parse_plan_steps,
    parse_plan_tensors,
    relocate_uops_words,
    words_to_uops_bytes,
)


@dataclass(frozen=True)
class PlanView:
    bundle: BundleH
    tensors: List[VcTensorDesc]
    steps: List[VcStepDesc]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _nchwc4_i8_size_bytes(n: int, c: int, h: int, w: int) -> int:
    c4 = _ceil_div(c, 4)
    return int(n) * int(c4) * int(h) * int(w) * 4


def _load_plan_view(bundle_h: Path) -> PlanView:
    text = bundle_h.read_text(encoding="utf-8")
    bundle = load_bundle_h(bundle_h)
    tensors = parse_plan_tensors(text)
    steps = parse_plan_steps(text)
    return PlanView(bundle=bundle, tensors=tensors, steps=steps)


def _span_bounds(tensors: List[VcTensorDesc]) -> Tuple[int, int]:
    if not tensors:
        return 0, 0
    base_min = min(int(t.offset_bytes) for t in tensors)
    end_max = max(int(t.offset_bytes) + int(t.size_bytes) for t in tensors)
    return base_min, end_max


def _check_tensor_table(plan: PlanView) -> None:
    b = plan.bundle
    tensors = plan.tensors
    arena_bytes = b.define_int("VC_PLAN_ARENA_BYTES")
    quant_scale_count = b.define_int("VC_PLAN_QUANT_SCALE_COUNT") if "VC_PLAN_QUANT_SCALE_COUNT" in b.defines_raw else 0
    address_mode_offset = b.define_int("ADDRESS_MODE_OFFSET") != 0

    if "VC_PLAN_TENSOR_COUNT" in b.defines_raw:
        expect = b.define_int("VC_PLAN_TENSOR_COUNT")
        if len(tensors) != expect:
            raise AssertionError(f"VC_TENSORS length mismatch: parsed={len(tensors)} macro={expect}")

    ids = [t.tensor_id for t in tensors]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate tensor_id found in VC_TENSORS")

    base_min, end_max = _span_bounds(tensors)
    span = end_max - base_min
    if span > arena_bytes:
        raise AssertionError(f"Tensor span exceeds arena: span={span} arena_bytes={arena_bytes}")

    for t in tensors:
        if t.quant_index != 0xFFFF and quant_scale_count:
            if int(t.quant_index) >= int(quant_scale_count):
                raise AssertionError(f"quant_index out of range for tensor_id={t.tensor_id}: {t.quant_index} >= {quant_scale_count}")

        expect_size = _nchwc4_i8_size_bytes(t.n, t.c, t.h, t.w)
        if int(t.size_bytes) != int(expect_size):
            raise AssertionError(
                f"size_bytes mismatch for tensor_id={t.tensor_id}: "
                f"size_bytes={t.size_bytes} expected={expect_size} (NCHWc4 int8)"
            )

        # Alignment: arena allocator uses 16B alignment.
        if address_mode_offset:
            if int(t.offset_bytes) % 16 != 0:
                raise AssertionError(f"offset_bytes not 16B aligned for tensor_id={t.tensor_id}: {t.offset_bytes}")
        else:
            if (int(t.offset_bytes) - int(base_min)) % 16 != 0:
                raise AssertionError(f"offset_bytes not 16B aligned (relative) for tensor_id={t.tensor_id}: {t.offset_bytes}")

        if int(t.size_bytes) == 0:
            raise AssertionError(f"Zero-sized tensor in plan: tensor_id={t.tensor_id}")


def _check_step_table(plan: PlanView) -> None:
    b = plan.bundle
    steps = plan.steps
    tensors = {t.tensor_id: t for t in plan.tensors}

    uops_len_words = b.define_int("VC_PLAN_UOPS_LEN_WORDS")
    params_len_words = b.define_int("VC_PLAN_PARAMS_LEN_WORDS")

    if "VC_PLAN_STEP_COUNT" in b.defines_raw:
        expect = b.define_int("VC_PLAN_STEP_COUNT")
        if len(steps) != expect:
            raise AssertionError(f"VC_STEPS length mismatch: parsed={len(steps)} macro={expect}")

    def _check_tensor_id(tid: int) -> None:
        if tid == 0xFFFF:
            return
        if tid not in tensors:
            raise AssertionError(f"Step references unknown tensor_id={tid}")

    uop_ranges: List[Tuple[int, int]] = []
    param_ranges: List[Tuple[int, int]] = []

    for idx, s in enumerate(steps):
        if int(s.step_type) not in (0, 1, 2):
            raise AssertionError(f"Invalid step_type at step[{idx}]: {s.step_type}")
        if int(s.input_count) > 4 or int(s.output_count) > 2:
            raise AssertionError(f"IO count out of range at step[{idx}]: in={s.input_count} out={s.output_count}")

        # Check input/output id padding.
        for j, tid in enumerate(s.inputs):
            if j < int(s.input_count):
                if tid == 0xFFFF:
                    raise AssertionError(f"Missing input tensor id at step[{idx}] input[{j}]")
                _check_tensor_id(tid)
            else:
                if tid != 0xFFFF:
                    raise AssertionError(f"Non-0xFFFF padding in inputs at step[{idx}] input[{j}]={tid}")

        for j, tid in enumerate(s.outputs):
            if j < int(s.output_count):
                if tid == 0xFFFF:
                    raise AssertionError(f"Missing output tensor id at step[{idx}] output[{j}]")
                _check_tensor_id(tid)
            else:
                if tid != 0xFFFF:
                    raise AssertionError(f"Non-0xFFFF padding in outputs at step[{idx}] output[{j}]={tid}")

        if int(s.step_type) == 1:
            # CPU step: uops/params segments must be empty.
            if any(int(x) != 0 for x in (s.uop_off_words, s.uop_words, s.param_off_words, s.param_words)):
                raise AssertionError(f"CPU step must have empty uops/params segment at step[{idx}]")
            # Kernel-specific invariants.
            if int(s.cpu_kernel) == 1:
                # CONCAT_C: v1 axis must be C (1).
                if int(s.axis) != 1:
                    raise AssertionError(f"CPU CONCAT_C requires axis==1 at step[{idx}] (got {s.axis})")
        else:
            # NPU/ALIAS steps may carry segments; ALIAS typically has empty segments but do not enforce.
            if int(s.uop_words) != 0:
                begin = int(s.uop_off_words)
                end = begin + int(s.uop_words)
                if begin < 0 or end > int(uops_len_words):
                    raise AssertionError(f"uops segment out of range at step[{idx}]: [{begin},{end}) > {uops_len_words}")
                uop_ranges.append((begin, end))
            if int(s.param_words) != 0:
                begin = int(s.param_off_words)
                end = begin + int(s.param_words)
                if begin < 0 or end > int(params_len_words):
                    raise AssertionError(f"params segment out of range at step[{idx}]: [{begin},{end}) > {params_len_words}")
                param_ranges.append((begin, end))

    def _check_non_overlapping(ranges: Iterable[Tuple[int, int]], label: str) -> None:
        rs = sorted(list(ranges))
        for (a0, a1), (b0, b1) in zip(rs, rs[1:]):
            if b0 < a1:
                raise AssertionError(f"{label} segments overlap: [{a0},{a1}) and [{b0},{b1})")

    _check_non_overlapping(uop_ranges, "uops")
    _check_non_overlapping(param_ranges, "params")


def _check_uop_address_ranges(plan: PlanView, act_base: int, param_base: int) -> None:
    b = plan.bundle
    address_mode_offset = b.define_int("ADDRESS_MODE_OFFSET") != 0

    arena_bytes = b.define_int("VC_PLAN_ARENA_BYTES")
    params_len_bytes = b.define_int("PARAMS_LEN_BYTES") if "PARAMS_LEN_BYTES" in b.defines_raw else (b.define_int("PARAMS_LEN_WORDS") * 4)
    param_base_macro = b.define_int("PARAM_BASE") if "PARAM_BASE" in b.defines_raw else 0

    # uops_words holds 6x u32 words per uOP.
    uops = decode_uops(words_to_uops_bytes(plan.bundle.uops_words))
    if not uops:
        raise AssertionError("No uOPs decoded from uops_words[]")

    # Determine activation address span used by plan tensors.
    base_min, end_max = _span_bounds(plan.tensors)
    if address_mode_offset:
        act_lo = 0
        act_hi = arena_bytes
        param_lo = 0
        param_hi = params_len_bytes
    else:
        act_lo = base_min
        act_hi = base_min + arena_bytes
        # Prefer PARAM_BASE macro when present (absolute mode).
        param_lo = param_base_macro
        param_hi = param_base_macro + params_len_bytes

    def _check_range(name: str, value: int, lo: int, hi: int) -> None:
        if not (lo <= int(value) < int(hi)):
            raise AssertionError(f"uOP {name} out of range: {value} not in [{lo},{hi})")

    for u in uops:
        _check_range("FI_ADDR", int(u.fi_addr), act_lo, act_hi)
        _check_range("FO_ADDR", int(u.fo_addr), act_lo, act_hi)
        _check_range("PARAM_ADDR", int(u.param_addr), param_lo, param_hi)

    # Optional: validate relocated absolute addresses in offset mode.
    if address_mode_offset:
        relocated = relocate_uops_words(plan.bundle.uops_words, act_base=act_base, param_base=param_base)
        uops2 = decode_uops(words_to_uops_bytes(relocated))
        act_lo2 = act_base
        act_hi2 = act_base + arena_bytes
        param_lo2 = param_base
        param_hi2 = param_base + params_len_bytes
        for u in uops2:
            _check_range("FI_ADDR(reloc)", int(u.fi_addr), act_lo2, act_hi2)
            _check_range("FO_ADDR(reloc)", int(u.fo_addr), act_lo2, act_hi2)
            _check_range("PARAM_ADDR(reloc)", int(u.param_addr), param_lo2, param_hi2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Host-side sanity checks for bundle plan/tensors/steps.")
    p.add_argument("--out-dir", type=Path, default=None, help="Compiler output directory (contains bundle.h).")
    p.add_argument("--bundle-h", type=Path, default=None, help="Path to bundle.h (overrides --out-dir).")
    p.add_argument("--act-base", type=lambda s: int(s, 0), default=0x2000_0000, help="Activation base for offset-mode relocation.")
    p.add_argument("--param-base", type=lambda s: int(s, 0), default=0x2100_0000, help="Param base for offset-mode relocation.")
    p.add_argument("--skip-uop-range-check", action="store_true", help="Skip uOP address range checks.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.bundle_h is None and args.out_dir is None:
        raise SystemExit("Provide --out-dir or --bundle-h")

    bundle_h = args.bundle_h or (args.out_dir / "bundle.h")
    if not bundle_h.exists():
        raise FileNotFoundError(f"bundle.h not found: {bundle_h}")

    plan = _load_plan_view(bundle_h)
    _check_tensor_table(plan)
    _check_step_table(plan)
    if not args.skip_uop_range_check:
        _check_uop_address_ranges(plan, act_base=args.act_base, param_base=args.param_base)

    print("[OK] plan sanity checks passed.")
    print(f"     bundle_h={bundle_h}")
    print(f"     tensors={len(plan.tensors)} steps={len(plan.steps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

