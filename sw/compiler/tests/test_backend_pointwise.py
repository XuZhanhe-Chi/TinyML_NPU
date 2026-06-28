# -*- coding: utf-8 -*-
"""
End-to-end smoke test for PointwiseConv (1x1) pipeline.
"""

from __future__ import annotations

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.config import HwConfig
from venuscore_compiler.ir.tensor import VcTensor
from venuscore_compiler.ir.ops import VcPointwiseConv
from venuscore_compiler.ir.program import VcProgram


def _make_const(shape, fill):
    data = []
    for _ in range(shape[0]):  # N or Cout
        dim1 = []
        for _ in range(shape[1]):
            dim2 = []
            for _ in range(shape[2]):
                row = [fill for _ in range(shape[3])]
                dim2.append(row)
            dim1.append(dim2)
        data.append(dim1)
    return data


def test_pointwise_conv_end_to_end(tmp_path: Path) -> None:
    """Compile a simple 1x1 conv and ensure artifacts exist."""

    program = VcProgram("pw_test")
    ifm = VcTensor(name="ifm", shape=(1, 8, 4, 4), layout="NCHW", dtype="int8")
    ofm = VcTensor(name="ofm", shape=(1, 8, 4, 4), layout="NCHW", dtype="int8")
    weight = VcTensor(name="weight", shape=(8, 8, 1, 1), layout="NCHW", dtype="int8", data=_make_const((8, 8, 1, 1), 1))
    bias = VcTensor(name="bias", shape=(1, 8, 1, 1), layout="NCHW", dtype="int32", data=_make_const((1, 8, 1, 1), 0))
    program.add_tensor(ifm)
    program.add_tensor(ofm)
    program.add_tensor(weight)
    program.add_tensor(bias)

    op = VcPointwiseConv(
        name="pw0",
        inputs=["ifm"],
        outputs=["ofm"],
        weight="weight",
        bias="bias",
        activation="none",
    )
    program.add_op(op)

    # Use a single-cluster config to avoid H-tiling for cluster balancing in this smoke test.
    hw = HwConfig(cluster_count=1, wbuf_lane_bytes=(1 << 16), wbuf_lanes=4, ibuf_line_bytes=3840, max_h_tile=255)
    artifact = compile_program(program, output_dir=tmp_path, dump_ir=False, dump_uop=False, hw=hw)

    assert (tmp_path / "uops.bin").exists()
    assert (tmp_path / "params.bin").exists()
    assert len(artifact.uops) == 1
    assert len(artifact.uop_binary) == 32  # single uOP (8x u32)
