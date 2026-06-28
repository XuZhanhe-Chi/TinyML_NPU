# -*- coding: utf-8 -*-
"""
Host-side driver self-check (unit test).

This test:
- Compiles a tiny program in offset addressing mode
- Parses the generated bundle.h
- Verifies uops_words[] matches uops.bin
- Verifies relocation (W3/W4/W5) behavior matches the driver contract
"""

from __future__ import annotations

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.frontend.manual_builder import build_single_conv3x3_program
from venuscore_compiler.isa.decoder import decode_uops
from venuscore_compiler.runtime.bundle_h_parser import (
    load_bundle_h,
    relocate_uops_words,
    words_to_uops_bytes,
)


def test_bundle_h_parse_and_relocation(tmp_path: Path) -> None:
    program = build_single_conv3x3_program(
        hin=4,
        win=4,
        cin=1,
        cout=4,
        padding=(0, 0, 0, 0),
        stride=(1, 1),
    )
    # Provide deterministic weights/bias (nested lists, no numpy dependency).
    weight = program.tensors["weight"]
    bias = program.tensors["bias"]
    kernel_3x3 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    weight.data = []
    for _ in range(4):  # Cout
        weight.data.append([kernel_3x3])  # Cin=1
    bias.data = [0, 0, 0, 0]

    # Offset mode: driver must relocate W3/W4/W5 at runtime.
    compile_program(
        program,
        output_dir=tmp_path,
        dump_ir=False,
        dump_uop=False,
        ifm_base=None,
        ofm_base=None,
        param_base=None,
    )

    bundle = load_bundle_h(tmp_path / "bundle.h")
    assert bundle.define_int("ADDRESS_MODE_OFFSET") == 1

    uops_bytes_from_h = words_to_uops_bytes(bundle.uops_words)
    uops_bytes_from_bin = (tmp_path / "uops.bin").read_bytes()
    assert uops_bytes_from_h == uops_bytes_from_bin

    uops_before = decode_uops(uops_bytes_from_h)
    assert uops_before

    act_base = 0x2000_0000
    param_base = 0x2100_0000
    relocated_words = relocate_uops_words(bundle.uops_words, act_base=act_base, param_base=param_base)
    uops_after = decode_uops(words_to_uops_bytes(relocated_words))

    assert len(uops_before) == len(uops_after)
    for a, b in zip(uops_before, uops_after):
        assert b.param_addr == (a.param_addr + param_base) & 0xFFFF_FFFF
        assert b.fi_addr == (a.fi_addr + act_base) & 0xFFFF_FFFF
        assert b.fo_addr == (a.fo_addr + act_base) & 0xFFFF_FFFF

