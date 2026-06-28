# -*- coding: utf-8 -*-
"""
Behavioral simulator smoke test against compiler outputs.

This test compiles a tiny Conv3x3 program and runs the uOP stream through the
functional simulator in sim/behavioral, verifying that output bytes match an
easily-checkable expected pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from venuscore_compiler import compile_program
from venuscore_compiler.frontend.manual_builder import build_single_conv3x3_program


def _load_sim():
    # sim/ is not part of the venuscore_compiler package; import by module path.
    from sim.behavioral.venuscore_sim import SparseWordMemory, SimConfig, VenusCoreSim, load_uops

    return SparseWordMemory, SimConfig, VenusCoreSim, load_uops


def _pack_nchw_to_nchwc4_bytes(value: int, h: int, w: int) -> bytes:
    """
    Build an NCHWc4 buffer for N=1,C=1 filled with 'value' in channel0.

    For C=1, each pixel word is: [value, 0, 0, 0].
    """
    b = value & 0xFF
    return bytes([b, 0, 0, 0] * (h * w))


def test_behavioral_sim_conv3x3_end_to_end(tmp_path: Path) -> None:
    """
    Compile Conv3x3 (Cin=1,Cout=4) and run it in the behavioral simulator.

    Input: all ones (channel0).
    Weights: all ones.
    Bias: zero.
    Expect: each output element equals 9 (3x3 sum), for all 4 output channels.
    """
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
    # weight shape: (Cout, Cin, Kh, Kw)
    kernel_3x3 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    weight.data = []
    for _ in range(4):  # Cout
        weight.data.append([kernel_3x3])  # Cin=1
    # bias shape can be [Cout]
    bias.data = [0, 0, 0, 0]

    artifact = compile_program(program, output_dir=tmp_path, dump_ir=False, dump_uop=False)

    meta_core = artifact.metadata
    output_base = int(meta_core.get("output_base", 0))
    output_size = int(meta_core.get("output_size", 0))
    assert output_size == 16  # Cout=4,H=2,W=2 => c4=1, 1*2*2*4 bytes

    SparseWordMemory, SimConfig, VenusCoreSim, load_uops = _load_sim()
    mem = SparseWordMemory()
    sim = VenusCoreSim(SimConfig(), mem)

    # Load params.bin at PARAM_ADDR.
    uops = load_uops(str(tmp_path / "uops.bin"))
    assert uops
    param_base = min(int(u.param_addr) for u in uops)
    mem.load_bytes(param_base, (tmp_path / "params.bin").read_bytes())

    # Load IFM at FI_ADDR (NCHWc4 bytes for C=1).
    fi_base = min(int(u.fi_addr) for u in uops)
    mem.load_bytes(fi_base, _pack_nchw_to_nchwc4_bytes(1, h=4, w=4))

    sim.run(uops, verbose=False)

    out = mem.dump_bytes(output_base, output_size)
    assert out == bytes([9] * output_size)
