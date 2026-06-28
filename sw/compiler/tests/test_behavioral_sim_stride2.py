# -*- coding: utf-8 -*-
"""
Behavioral simulator regression tests for stride=2.

These tests ensure the uOP-level functional simulator matches the ISA semantics
for stride=2 on Conv2D and DWCONV:
  - FI_ADDR / FO_ADDR are channel-plane bases (no Y-offset baked in)
  - Y_INDEX + STRIDE + PAD_* drive the sampling positions
"""

from __future__ import annotations

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.frontend.manual_builder import build_single_conv3x3_program
from venuscore_compiler.ir.ops import VcDepthwiseConv
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.ir.tensor import VcTensor


def _load_sim():
    from sim.behavioral.venuscore_sim import SparseWordMemory, SimConfig, VenusCoreSim, load_uops

    return SparseWordMemory, SimConfig, VenusCoreSim, load_uops


def _pack_nchw_to_nchwc4_bytes_const(ch_values: tuple[int, int, int, int], h: int, w: int) -> bytes:
    b0, b1, b2, b3 = (v & 0xFF for v in ch_values)
    return bytes([b0, b1, b2, b3] * (h * w))


def test_behavioral_sim_conv3x3_stride2(tmp_path: Path) -> None:
    # 4x4 input, 3x3 kernel, stride=2 -> output 1x1
    program = build_single_conv3x3_program(
        hin=4,
        win=4,
        cin=1,
        cout=4,
        padding=(0, 0, 0, 0),
        stride=(2, 2),
    )

    # Deterministic weights: all ones on Cin=1 (other Cin lanes padded to 0 by packer)
    weight = program.tensors["weight"]
    bias = program.tensors["bias"]
    kernel_3x3 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    weight.data = []
    for _ in range(4):  # Cout
        weight.data.append([kernel_3x3])  # Cin=1
    bias.data = [0, 0, 0, 0]

    artifact = compile_program(program, output_dir=tmp_path, dump_ir=False, dump_uop=False)

    SparseWordMemory, SimConfig, VenusCoreSim, load_uops = _load_sim()
    mem = SparseWordMemory()
    sim = VenusCoreSim(SimConfig(), mem)

    uops = load_uops(str(tmp_path / "uops.bin"))
    assert uops

    # Load params at PARAM_ADDR
    param_base = min(int(u.param_addr) for u in uops)
    mem.load_bytes(param_base, (tmp_path / "params.bin").read_bytes())

    # Load IFM: C=1 -> each pixel word is [1,0,0,0]
    fi_base = min(int(u.fi_addr) for u in uops)
    mem.load_bytes(fi_base, _pack_nchw_to_nchwc4_bytes_const((1, 0, 0, 0), h=4, w=4))

    sim.run(uops, verbose=False)

    meta = artifact.metadata
    output_base = int(meta.get("output_base", 0))
    output_size = int(meta.get("output_size", 0))
    assert output_size == 4  # Cout=4,H=1,W=1 => one word
    out = mem.dump_bytes(output_base, output_size)
    assert out == bytes([9, 9, 9, 9])


def test_behavioral_sim_dw3x3_stride2(tmp_path: Path) -> None:
    hin = win = 4
    cin = cout = 4
    padding = (0, 0, 0, 0)
    stride = (2, 2)
    hout = (hin - 3) // 2 + 1
    wout = (win - 3) // 2 + 1
    assert (hout, wout) == (1, 1)

    program = VcProgram(name="dw_stride2_sim")
    program.add_tensor(VcTensor(name="input", shape=(1, cin, hin, win), dtype="int8"))
    program.add_tensor(VcTensor(name="output", shape=(1, cout, hout, wout), dtype="int8"))
    # ONNX-style DW weights: [Cin,1,3,3]
    program.add_tensor(VcTensor(name="weight", shape=(cin, 1, 3, 3), dtype="int8"))
    program.add_tensor(VcTensor(name="bias", shape=(1, cout, 1, 1), dtype="int32"))

    program.add_op(
        VcDepthwiseConv(
            name="dw0",
            inputs=["input"],
            outputs=["output"],
            weight="weight",
            bias="bias",
            kernel=(3, 3),
            stride=stride,
            padding=padding,
            groups=cin,
            activation="relu",
            qmode=None,
        )
    )
    program.validate()

    # Fill weight/bias data
    weight = program.tensors["weight"]
    bias = program.tensors["bias"]
    weight.data = [[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]] * cin
    bias.data = [0, 0, 0, 0]

    artifact = compile_program(program, output_dir=tmp_path, dump_ir=False, dump_uop=False)

    SparseWordMemory, SimConfig, VenusCoreSim, load_uops = _load_sim()
    mem = SparseWordMemory()
    sim = VenusCoreSim(SimConfig(), mem)

    uops = load_uops(str(tmp_path / "uops.bin"))
    assert uops
    param_base = min(int(u.param_addr) for u in uops)
    mem.load_bytes(param_base, (tmp_path / "params.bin").read_bytes())

    fi_base = min(int(u.fi_addr) for u in uops)
    mem.load_bytes(fi_base, _pack_nchw_to_nchwc4_bytes_const((1, 1, 1, 1), h=4, w=4))

    sim.run(uops, verbose=False)

    meta = artifact.metadata
    output_base = int(meta.get("output_base", 0))
    output_size = int(meta.get("output_size", 0))
    assert output_size == 4
    out = mem.dump_bytes(output_base, output_size)
    assert out == bytes([9, 9, 9, 9])

