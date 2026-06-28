# -*- coding: utf-8 -*-
"""
End-to-end smoke test for AvgPool pipeline.
"""

from __future__ import annotations

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.config import HwConfig
from venuscore_compiler.ir.tensor import VcTensor
from venuscore_compiler.ir.ops import VcAvgPool
from venuscore_compiler.ir.program import VcProgram


def _make_const(shape, fill):
    data = []
    for _ in range(shape[0]):  # N
        dim1 = []
        for _ in range(shape[1]):  # C
            dim2 = []
            for _ in range(shape[2]):  # H
                row = [fill for _ in range(shape[3])]  # W
                dim2.append(row)
            dim1.append(dim2)
        data.append(dim1)
    return data


def test_avgpool_end_to_end(tmp_path: Path) -> None:
    """Compile a simple 2x2 avgpool and ensure artifacts exist."""

    program = VcProgram("avgpool_test")
    ifm = VcTensor(name="ifm", shape=(1, 4, 4, 4), layout="NCHW", dtype="int8", data=_make_const((1, 4, 4, 4), 1))
    ofm = VcTensor(name="ofm", shape=(1, 4, 2, 2), layout="NCHW", dtype="int8")
    program.add_tensor(ifm)
    program.add_tensor(ofm)

    op = VcAvgPool(
        name="pool0",
        inputs=["ifm"],
        outputs=["ofm"],
        kernel=(2, 2),
        stride=(2, 2),
        padding=(0, 0, 0, 0),
    )
    program.add_op(op)

    # Use a single-cluster config to avoid H-tiling for cluster balancing in this smoke test.
    hw = HwConfig(cluster_count=1, wbuf_lane_bytes=(1 << 16), wbuf_lanes=4, ibuf_line_bytes=3840, max_h_tile=255)
    artifact = compile_program(program, output_dir=tmp_path, dump_ir=False, dump_uop=False, hw=hw)

    assert (tmp_path / "uops.bin").exists()
    assert (tmp_path / "params.bin").exists()
    assert len(artifact.uops) == 1
    # AvgPool has no weights, but it still emits the Quant Coeff Block (bias/scale/shift)
    # for SFU requantization: Cout * 8 bytes.
    assert len(artifact.param_block) == 4 * 8
    # With no quant info on tensors, we default to scale=1, shift=0, bias=0 for all channels.
    assert artifact.param_block == (b"\x00\x00\x00\x00\x01\x00\x00\x00" * 4)
