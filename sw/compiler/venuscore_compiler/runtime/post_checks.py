# -*- coding: utf-8 -*-
"""
Post-compile host-side checks.

These checks are designed to catch integration issues before board/FPGA bring-up:
- bundle.h <-> uops.bin/params.bin consistency
- uOP decoding consistency between compiler ISA decoder and behavioral sim decoder
- offset-mode relocation contract checks (W3/W4/W5)
- basic Plan sanity checks (tensor/step bounds, segment bounds, address ranges)

The compiler can call these checks automatically after emitting artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from venuscore_compiler.isa.decoder import decode_uops
from venuscore_compiler.isa.layout_spec import Activation
from venuscore_compiler.isa.uop_format import UOP_SIZE_BYTES
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


def _load_sim_decoder():
    # sim/ is not part of the venuscore_compiler package; import by module path.
    try:
        from sim.behavioral.venuscore_sim import Uop as SimUop
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Post-check requires importing sim.behavioral.venuscore_sim. "
            "If you are using venuscore_compiler as a standalone package, "
            "disable post_check or run checks via scripts inside the repo."
        ) from e

    return SimUop


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _nchwc4_i8_size_bytes(n: int, c: int, h: int, w: int) -> int:
    c4 = _ceil_div(c, 4)
    return int(n) * int(c4) * int(h) * int(w) * 4


def _span_bounds(tensors: List[VcTensorDesc]) -> Tuple[int, int]:
    if not tensors:
        return 0, 0
    base_min = min(int(t.offset_bytes) for t in tensors)
    end_max = max(int(t.offset_bytes) + int(t.size_bytes) for t in tensors)
    return base_min, end_max


def _check_non_overlapping(ranges: Iterable[Tuple[int, int]], label: str) -> None:
    rs = sorted(list(ranges))
    for (a0, a1), (b0, b1) in zip(rs, rs[1:]):
        if b0 < a1:
            raise AssertionError(f"{label} segments overlap: [{a0},{a1}) and [{b0},{b1})")


def _compare_uop_decoders(uops_bytes: bytes) -> None:
    SimUop = _load_sim_decoder()
    uops_a = decode_uops(uops_bytes)
    if len(uops_bytes) % UOP_SIZE_BYTES != 0:
        raise ValueError(f"uOP stream length is not a multiple of {UOP_SIZE_BYTES} bytes")
    uops_b = [SimUop.decode_bytes(uops_bytes, off) for off in range(0, len(uops_bytes), UOP_SIZE_BYTES)]

    if len(uops_a) != len(uops_b):
        raise AssertionError(f"uOP count mismatch: compiler={len(uops_a)} sim={len(uops_b)}")

    for idx, (a, b) in enumerate(zip(uops_a, uops_b)):
        act_expected = int(a.act.value) if a.act is not None else int(Activation.NONE.value)
        if int(b.opcode) != int(a.opcode.value):
            raise AssertionError(f"[decode] opcode mismatch at {idx}: compiler={a.opcode} sim={b.opcode}")
        if int(b.act_type) != int(act_expected):
            raise AssertionError(f"[decode] act mismatch at {idx}: compiler={a.act} sim_act_type={b.act_type}")
        if int(b.first) != int(bool(a.first_flag)):
            raise AssertionError(f"[decode] first_flag mismatch at {idx}")
        if int(b.last) != int(bool(a.last_flag)):
            raise AssertionError(f"[decode] last_flag mismatch at {idx}")
        if int(b.sync) != int(bool(a.sync)):
            raise AssertionError(f"[decode] sync mismatch at {idx}")

        if int(b.h_tile) != int(a.h_tile) or int(b.w_tile) != int(a.w_tile):
            raise AssertionError(f"[decode] tile geom mismatch at {idx}")
        if int(b.c4_in) != int(a.c4_in) or int(b.c4_out) != int(a.c4_out):
            raise AssertionError(f"[decode] c4 mismatch at {idx}")
        if int(b.y_index) != int(a.y_index):
            raise AssertionError(f"[decode] y_index mismatch at {idx}")

        if int(b.ifm_w) != int(a.ifm_w) or int(b.ifm_h) != int(a.ifm_h):
            raise AssertionError(f"[decode] ifm dims mismatch at {idx}: compiler=({a.ifm_h},{a.ifm_w}) sim=({b.ifm_h},{b.ifm_w})")
        if int(b.actdma_line_words) != int(a.actdma_line_words):
            raise AssertionError(f"[decode] actdma_line_words mismatch at {idx}: compiler={a.actdma_line_words} sim={b.actdma_line_words}")
        if int(b.outdma_line_words) != int(a.outdma_line_words):
            raise AssertionError(f"[decode] outdma_line_words mismatch at {idx}: compiler={a.outdma_line_words} sim={b.outdma_line_words}")

        if int(b.stride()) != int(a.stride_h):
            raise AssertionError(f"[decode] stride mismatch at {idx}: compiler={a.stride_h} sim={b.stride()}")
        pt, pb, pl, pr = b.pads()
        if (pt, pb, pl, pr) != (a.pad_top, a.pad_bottom, a.pad_left, a.pad_right):
            raise AssertionError(f"[decode] pads mismatch at {idx}")

        if int(b.fi_stride) != int(a.fi_stride) or int(b.fo_stride) != int(a.fo_stride):
            raise AssertionError(f"[decode] stride words mismatch at {idx}")

        if int(b.param_addr) != int(a.param_addr) or int(b.fi_addr) != int(a.fi_addr) or int(b.fo_addr) != int(a.fo_addr):
            raise AssertionError(f"[decode] addr mismatch at {idx}")


def run_bundle_relocation_checks(out_dir: Path, act_base: int, param_base: int) -> None:
    """
    Validate:
    - bundle.h arrays match uops.bin/params.bin
    - uOP decoding matches compiler decoder and sim decoder
    - offset relocation contract is correct (if enabled)
    """
    out_dir = Path(out_dir)
    bundle_h = out_dir / "bundle.h"
    uops_bin = out_dir / "uops.bin"
    params_bin = out_dir / "params.bin"

    if not bundle_h.exists():
        raise FileNotFoundError(f"bundle.h not found: {bundle_h}")
    bundle = load_bundle_h(bundle_h)

    address_mode_offset = bundle.define_int("ADDRESS_MODE_OFFSET") != 0
    uops_len_words = bundle.define_int("UOPS_LEN_WORDS")
    params_len_words = bundle.define_int("PARAMS_LEN_WORDS")

    if len(bundle.uops_words) != uops_len_words:
        raise AssertionError(f"uops_words length mismatch: header={len(bundle.uops_words)} macro={uops_len_words}")
    if len(bundle.params_words) != params_len_words:
        raise AssertionError(f"params_words length mismatch: header={len(bundle.params_words)} macro={params_len_words}")

    uops_from_h = words_to_uops_bytes(bundle.uops_words)
    _compare_uop_decoders(uops_from_h)

    if uops_bin.exists():
        uops_from_bin = uops_bin.read_bytes()
        if uops_from_bin != uops_from_h:
            raise AssertionError("uops.bin does not match uops_words[] in bundle.h")

    if params_bin.exists():
        params_from_h = b"".join(int(w).to_bytes(4, "little", signed=False) for w in bundle.params_words)
        params_from_bin = params_bin.read_bytes()
        if params_from_bin != params_from_h[: len(params_from_bin)]:
            raise AssertionError("params.bin does not match params_words[] in bundle.h (prefix compare)")

    if address_mode_offset:
        uops_a = decode_uops(uops_from_h)
        relocated_words = relocate_uops_words(bundle.uops_words, act_base=act_base, param_base=param_base)
        uops_b = decode_uops(words_to_uops_bytes(relocated_words))
        if len(uops_a) != len(uops_b):
            raise AssertionError("uOP count mismatch after relocation")
        for idx, (a, b) in enumerate(zip(uops_a, uops_b)):
            if int(b.param_addr) != (int(a.param_addr) + int(param_base)) & 0xFFFF_FFFF:
                raise AssertionError(f"[reloc] param_addr mismatch at {idx}")
            if int(b.fi_addr) != (int(a.fi_addr) + int(act_base)) & 0xFFFF_FFFF:
                raise AssertionError(f"[reloc] fi_addr mismatch at {idx}")
            if int(b.fo_addr) != (int(a.fo_addr) + int(act_base)) & 0xFFFF_FFFF:
                raise AssertionError(f"[reloc] fo_addr mismatch at {idx}")


@dataclass(frozen=True)
class PlanView:
    bundle: BundleH
    tensors: List[VcTensorDesc]
    steps: List[VcStepDesc]


def _load_plan_view(bundle_h: Path) -> PlanView:
    text = Path(bundle_h).read_text(encoding="utf-8")
    bundle = load_bundle_h(Path(bundle_h))
    tensors = parse_plan_tensors(text)
    steps = parse_plan_steps(text)
    return PlanView(bundle=bundle, tensors=tensors, steps=steps)


def run_plan_sanity_checks(bundle_h: Path, act_base: int, param_base: int, *, skip_uop_range_check: bool = False) -> None:
    """
    Validate plan tables and address ranges for board/FPGA integration.

    Raises AssertionError/ValueError on failure.
    """
    bundle_h = Path(bundle_h)
    if not bundle_h.exists():
        raise FileNotFoundError(f"bundle.h not found: {bundle_h}")

    text = bundle_h.read_text(encoding="utf-8")
    defines_raw = parse_defines(text)
    if "VC_HAS_PLAN" in defines_raw:
        if int(defines_raw["VC_HAS_PLAN"].strip().rstrip("u").rstrip("U") or "0", 0) == 0:
            return

    plan = _load_plan_view(bundle_h)
    b = plan.bundle
    tensors = plan.tensors
    steps = plan.steps

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

        if address_mode_offset:
            if int(t.offset_bytes) % 16 != 0:
                raise AssertionError(f"offset_bytes not 16B aligned for tensor_id={t.tensor_id}: {t.offset_bytes}")
        else:
            if (int(t.offset_bytes) - int(base_min)) % 16 != 0:
                raise AssertionError(f"offset_bytes not 16B aligned (relative) for tensor_id={t.tensor_id}: {t.offset_bytes}")

        if int(t.size_bytes) == 0:
            raise AssertionError(f"Zero-sized tensor in plan: tensor_id={t.tensor_id}")

    uops_len_words = b.define_int("VC_PLAN_UOPS_LEN_WORDS")
    params_len_words = b.define_int("VC_PLAN_PARAMS_LEN_WORDS")

    if "VC_PLAN_STEP_COUNT" in b.defines_raw:
        expect = b.define_int("VC_PLAN_STEP_COUNT")
        if len(steps) != expect:
            raise AssertionError(f"VC_STEPS length mismatch: parsed={len(steps)} macro={expect}")

    tensor_map: Dict[int, VcTensorDesc] = {t.tensor_id: t for t in tensors}

    def _check_tensor_id(tid: int) -> None:
        if tid == 0xFFFF:
            return
        if tid not in tensor_map:
            raise AssertionError(f"Step references unknown tensor_id={tid}")

    uop_ranges: List[Tuple[int, int]] = []
    param_ranges: List[Tuple[int, int]] = []

    for idx, s in enumerate(steps):
        if int(s.step_type) not in (0, 1, 2):
            raise AssertionError(f"Invalid step_type at step[{idx}]: {s.step_type}")
        if int(s.input_count) > 4 or int(s.output_count) > 2:
            raise AssertionError(f"IO count out of range at step[{idx}]: in={s.input_count} out={s.output_count}")

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
            if any(int(x) != 0 for x in (s.uop_off_words, s.uop_words, s.param_off_words, s.param_words)):
                raise AssertionError(f"CPU step must have empty uops/params segment at step[{idx}]")
            if int(s.cpu_activation) not in (0, 1, 2):
                raise AssertionError(f"Invalid cpu_activation at step[{idx}]: {s.cpu_activation}")
            if int(s.cpu_kernel) == 1:
                if int(s.axis) != 1:
                    raise AssertionError(f"CPU CONCAT_C requires axis==1 at step[{idx}] (got {s.axis})")
        else:
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

    _check_non_overlapping(uop_ranges, "uops")
    _check_non_overlapping(param_ranges, "params")

    if skip_uop_range_check:
        return

    params_len_bytes = b.define_int("PARAMS_LEN_BYTES") if "PARAMS_LEN_BYTES" in b.defines_raw else (b.define_int("PARAMS_LEN_WORDS") * 4)
    param_base_macro = b.define_int("PARAM_BASE") if "PARAM_BASE" in b.defines_raw else 0

    uops = decode_uops(words_to_uops_bytes(b.uops_words))
    if not uops:
        raise AssertionError("No uOPs decoded from uops_words[]")

    if address_mode_offset:
        act_lo = 0
        act_hi = arena_bytes
        param_lo = 0
        param_hi = params_len_bytes
    else:
        act_lo = base_min
        act_hi = base_min + arena_bytes
        param_lo = param_base_macro
        param_hi = param_base_macro + params_len_bytes

    def _check_range(name: str, value: int, lo: int, hi: int) -> None:
        if not (lo <= int(value) < int(hi)):
            raise AssertionError(f"uOP {name} out of range: {value} not in [{lo},{hi})")

    for u in uops:
        _check_range("FI_ADDR", int(u.fi_addr), act_lo, act_hi)
        _check_range("FO_ADDR", int(u.fo_addr), act_lo, act_hi)
        _check_range("PARAM_ADDR", int(u.param_addr), param_lo, param_hi)

    if address_mode_offset:
        relocated = relocate_uops_words(b.uops_words, act_base=act_base, param_base=param_base)
        uops2 = decode_uops(words_to_uops_bytes(relocated))
        act_lo2 = act_base
        act_hi2 = act_base + arena_bytes
        param_lo2 = param_base
        param_hi2 = param_base + params_len_bytes
        for u in uops2:
            _check_range("FI_ADDR(reloc)", int(u.fi_addr), act_lo2, act_hi2)
            _check_range("FO_ADDR(reloc)", int(u.fo_addr), act_lo2, act_hi2)
            _check_range("PARAM_ADDR(reloc)", int(u.param_addr), param_lo2, param_hi2)


def run_post_compile_checks(
    out_dir: Path,
    *,
    act_base: int = 0x2000_0000,
    param_base: int = 0x2100_0000,
) -> None:
    """
    Run all host-side post-compile checks on artifacts under out_dir.
    """
    out_dir = Path(out_dir)
    run_bundle_relocation_checks(out_dir, act_base=act_base, param_base=param_base)
    run_plan_sanity_checks(out_dir / "bundle.h", act_base=act_base, param_base=param_base, skip_uop_range_check=False)
