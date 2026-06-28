# -*- coding: utf-8 -*-
"""
Host-side bundle/driver self-check.

What it validates:
1) bundle.h arrays match uops.bin/params.bin (word-for-word)
2) uOP decoding matches between:
   - venuscore_compiler.isa.decoder (used by scripts/dump_uop_debug.py)
   - sim.behavioral.venuscore_sim.Uop (behavioral simulator decoder)
3) If ADDRESS_MODE_OFFSET==1, relocation of W3/W4/W5 matches expectations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from venuscore_compiler.isa.decoder import decode_uops
from venuscore_compiler.isa.layout_spec import Activation
from venuscore_compiler.isa.uop_format import UOP_SIZE_BYTES
from venuscore_compiler.runtime.bundle_h_parser import (
    load_bundle_h,
    relocate_uops_words,
    words_to_uops_bytes,
)


def _load_sim_decoder():
    # sim/ is not part of venuscore_compiler package; import by module path.
    from sim.behavioral.venuscore_sim import Uop as SimUop

    return SimUop


def _compare_decoders(uops_bytes: bytes) -> None:
    SimUop = _load_sim_decoder()
    uops_a = decode_uops(uops_bytes)
    if len(uops_bytes) % UOP_SIZE_BYTES != 0:
        raise ValueError(f"uOP stream length is not a multiple of {UOP_SIZE_BYTES} bytes")
    uops_b = [SimUop.decode_bytes(uops_bytes, off) for off in range(0, len(uops_bytes), UOP_SIZE_BYTES)]

    if len(uops_a) != len(uops_b):
        raise AssertionError(f"uOP count mismatch: compiler={len(uops_a)} sim={len(uops_b)}")

    for idx, (a, b) in enumerate(zip(uops_a, uops_b)):
        act_expected = None
        if a.act is not None:
            act_expected = int(a.act.value)
        else:
            act_expected = int(Activation.NONE.value)
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
            raise AssertionError(f"[decode] ifm dims mismatch at {idx}")
        if int(b.actdma_line_words) != int(a.actdma_line_words):
            raise AssertionError(f"[decode] actdma_line_words mismatch at {idx}")
        if int(b.outdma_line_words) != int(a.outdma_line_words):
            raise AssertionError(f"[decode] outdma_line_words mismatch at {idx}")

        if int(b.stride()) != int(a.stride_h):
            raise AssertionError(f"[decode] stride mismatch at {idx}: compiler={a.stride_h} sim={b.stride()}")
        pt, pb, pl, pr = b.pads()
        if (pt, pb, pl, pr) != (a.pad_top, a.pad_bottom, a.pad_left, a.pad_right):
            raise AssertionError(f"[decode] pads mismatch at {idx}")

        if int(b.fi_stride) != int(a.fi_stride) or int(b.fo_stride) != int(a.fo_stride):
            raise AssertionError(f"[decode] stride words mismatch at {idx}")

        if int(b.param_addr) != int(a.param_addr) or int(b.fi_addr) != int(a.fi_addr) or int(b.fo_addr) != int(a.fo_addr):
            raise AssertionError(f"[decode] addr mismatch at {idx}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Host-side validation of bundle.h and relocation.")
    p.add_argument("--out-dir", type=Path, default=None, help="Compiler output directory (contains bundle.h/uops.bin/params.bin).")
    p.add_argument("--bundle-h", type=Path, default=None, help="Path to bundle.h (overrides --out-dir).")
    p.add_argument("--uops-bin", type=Path, default=None, help="Optional path to uops.bin (overrides --out-dir).")
    p.add_argument("--params-bin", type=Path, default=None, help="Optional path to params.bin (overrides --out-dir).")
    p.add_argument("--act-base", type=lambda s: int(s, 0), default=0x2000_0000, help="Activation base for relocation (offset mode).")
    p.add_argument("--param-base", type=lambda s: int(s, 0), default=0x2100_0000, help="Param base for relocation (offset mode).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.bundle_h is None and args.out_dir is None:
        raise SystemExit("Provide --out-dir or --bundle-h")

    out_dir = args.out_dir
    bundle_h = args.bundle_h or (out_dir / "bundle.h")
    uops_bin = args.uops_bin or (out_dir / "uops.bin" if out_dir else None)
    params_bin = args.params_bin or (out_dir / "params.bin" if out_dir else None)

    if bundle_h is None or not bundle_h.exists():
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
    _compare_decoders(uops_from_h)

    if uops_bin is not None and uops_bin.exists():
        uops_from_bin = uops_bin.read_bytes()
        if uops_from_bin != uops_from_h:
            raise AssertionError("uops.bin does not match uops_words[] in bundle.h")

    if params_bin is not None and params_bin.exists():
        # params_words is little-endian u32 words.
        params_from_h = b"".join(int(w).to_bytes(4, "little", signed=False) for w in bundle.params_words)
        params_from_bin = params_bin.read_bytes()
        if params_from_bin != params_from_h[: len(params_from_bin)]:
            raise AssertionError("params.bin does not match params_words[] in bundle.h (prefix compare)")

    if address_mode_offset:
        uops_a = decode_uops(uops_from_h)
        relocated_words = relocate_uops_words(bundle.uops_words, act_base=args.act_base, param_base=args.param_base)
        uops_b = decode_uops(words_to_uops_bytes(relocated_words))
        if len(uops_a) != len(uops_b):
            raise AssertionError("uOP count mismatch after relocation")
        for idx, (a, b) in enumerate(zip(uops_a, uops_b)):
            if int(b.param_addr) != (int(a.param_addr) + int(args.param_base)) & 0xFFFF_FFFF:
                raise AssertionError(f"[reloc] param_addr mismatch at {idx}")
            if int(b.fi_addr) != (int(a.fi_addr) + int(args.act_base)) & 0xFFFF_FFFF:
                raise AssertionError(f"[reloc] fi_addr mismatch at {idx}")
            if int(b.fo_addr) != (int(a.fo_addr) + int(args.act_base)) & 0xFFFF_FFFF:
                raise AssertionError(f"[reloc] fo_addr mismatch at {idx}")

    print("[OK] bundle.h parsing, decoding, and relocation checks passed.")
    print(f"     bundle_h={bundle_h}")
    print(f"     address_mode_offset={int(address_mode_offset)} uop_count={len(uops_from_h) // UOP_SIZE_BYTES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
