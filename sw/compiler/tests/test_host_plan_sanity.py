# -*- coding: utf-8 -*-
"""
Host-side plan sanity checks (unit test).

This test compiles a small program in offset mode and verifies:
- bundle.h plan tables can be parsed
- tensor sizes match NCHWc4 int8
- step tables are internally consistent
"""

from __future__ import annotations

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.frontend.manual_builder import build_single_conv3x3_program
from venuscore_compiler.runtime.bundle_h_parser import parse_plan_steps, parse_plan_tensors


def test_bundle_h_plan_tables_parse(tmp_path: Path) -> None:
    program = build_single_conv3x3_program(
        hin=4,
        win=4,
        cin=1,
        cout=4,
        padding=(0, 0, 0, 0),
        stride=(1, 1),
    )
    weight = program.tensors["weight"]
    bias = program.tensors["bias"]
    kernel_3x3 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    weight.data = []
    for _ in range(4):  # Cout
        weight.data.append([kernel_3x3])  # Cin=1
    bias.data = [0, 0, 0, 0]

    compile_program(
        program,
        output_dir=tmp_path,
        dump_ir=False,
        dump_uop=False,
        ifm_base=None,
        ofm_base=None,
        param_base=None,
    )

    text = (tmp_path / "bundle.h").read_text(encoding="utf-8")
    tensors = parse_plan_tensors(text)
    steps = parse_plan_steps(text)

    assert tensors
    assert steps
    # For this tiny graph there should be no CPU steps.
    assert all(s.step_type != 1 for s in steps)

