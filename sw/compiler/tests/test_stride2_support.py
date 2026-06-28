# -*- coding: utf-8 -*-
"""
Regression tests for stride=2 support in Conv2D and DepthwiseConv.

The VenusCore ISA encodes stride with a 2-bit STRIDE field:
  - 1 -> stride 1
  - 2 -> stride 2

These tests ensure the compiler pipeline can emit stride=2 uOPs and that
FI_ADDR/FO_ADDR remain channel-plane bases (no Y-offset baked into addresses).
"""

from __future__ import annotations

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.config import default_hw_config
from venuscore_compiler.frontend.manual_builder import build_single_conv3x3_program
from venuscore_compiler.ir.ops import VcDepthwiseConv
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.ir.tensor import VcTensor


def test_conv2d_stride2_uops_have_constant_plane_bases(tmp_path: Path) -> None:
    hw = default_hw_config()
    hw.cluster_count = 2  # encourage H-tiling to produce multiple uOPs

    program = build_single_conv3x3_program(
        hin=8,
        win=8,
        cin=4,
        cout=4,
        padding=(1, 1, 1, 1),
        stride=(2, 2),
    )

    artifact = compile_program(program, output_dir=tmp_path, hw=hw, dump_ir=False, dump_uop=False)
    assert artifact.uops

    # With H tiling, multiple uOPs should share the same FI/FO plane base.
    fi_addrs = {int(u.fi_addr) for u in artifact.uops}
    fo_addrs = {int(u.fo_addr) for u in artifact.uops}
    assert len(fi_addrs) == 1
    assert len(fo_addrs) == 1

    for u in artifact.uops:
        assert u.stride_h == 2 and u.stride_w == 2
        assert u.pad_top in (0, 1) and u.pad_bottom in (0, 1)


def test_dwconv_stride2_uops_have_constant_plane_bases(tmp_path: Path) -> None:
    hw = default_hw_config()
    hw.cluster_count = 2

    hin = win = 8
    cin = cout = 4
    padding = (1, 1, 1, 1)
    stride = (2, 2)
    pad_top, pad_bottom, pad_left, pad_right = padding
    sh, sw = stride
    hout = (hin + pad_top + pad_bottom - 3) // sh + 1
    wout = (win + pad_left + pad_right - 3) // sw + 1

    program = VcProgram(name="dw_stride2")
    program.add_tensor(
        VcTensor(
            name="input",
            shape=(1, cin, hin, win),
            dtype="int8",
            data=[[[[1 for _ in range(win)] for _ in range(hin)] for _ in range(cin)]],
        )
    )
    # ONNX-style DW weight layout [Cin, 1, 3, 3] is accepted by param block builder.
    program.add_tensor(
        VcTensor(
            name="weight",
            shape=(cin, 1, 3, 3),
            dtype="int8",
            data=[[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]] for _ in range(cin)],
        )
    )
    program.add_tensor(
        VcTensor(
            name="bias",
            shape=(1, cout, 1, 1),
            dtype="int32",
            data=[[[[0]] for _ in range(cout)]],
        )
    )
    program.add_tensor(VcTensor(name="output", shape=(1, cout, hout, wout), dtype="int8"))

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

    artifact = compile_program(program, output_dir=tmp_path, hw=hw, dump_ir=False, dump_uop=False)
    assert artifact.uops

    fi_addrs = {int(u.fi_addr) for u in artifact.uops}
    fo_addrs = {int(u.fo_addr) for u in artifact.uops}
    assert len(fi_addrs) == 1
    assert len(fo_addrs) == 1

    for u in artifact.uops:
        assert u.stride_h == 2 and u.stride_w == 2
        # Padding is tile-level: only the first output stripe has top padding.
        assert u.pad_left == 1 and u.pad_right == 1
        assert u.pad_top == (1 if u.y_index == 0 else 0)
        assert u.pad_bottom == (1 if (u.y_index + u.h_tile) >= hout else 0)
