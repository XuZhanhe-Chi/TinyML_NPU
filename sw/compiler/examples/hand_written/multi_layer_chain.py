# -*- coding: utf-8 -*-
"""
Multi-layer handwritten network to exercise ping-pong memory planning and uOP generation.

Network:
  layer0: Conv3x3 (Cin=16 -> Cout=24)
  layer1: PointwiseConv1x1 (Cin=24 -> Cout=24)

The goal is to produce multiple IR ops, run through the full pipeline, and
inspect the resulting uOP stream and memory usage.
"""

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.config import HwConfig
from venuscore_compiler.ir.ops import VcConv2D, VcPointwiseConv
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.ir.tensor import VcTensor
from venuscore_compiler.utils.debug_dump import dump_ir


def _make_input(n: int, c: int, h: int, w: int) -> list:
    data = []
    for _ in range(n):
        channels = []
        for _ in range(c):
            plane = []
            for y in range(h):
                row = []
                for x in range(w):
                    row.append((x + y) % 8)
                plane.append(row)
            channels.append(plane)
        data.append(channels)
    return data


def _make_weight(cout: int, cin: int, kh: int, kw: int) -> list:
    weights = []
    for co in range(cout):
        cin_list = []
        for ci in range(cin):
            kernel = []
            for y in range(kh):
                row = []
                for x in range(kw):
                    row.append(((co + ci + x + y) % 5) - 2)
                kernel.append(row)
            cin_list.append(kernel)
        weights.append(cin_list)
    return weights


def _make_bias(cout: int) -> list:
    return [[[[0] for _ in range(1)] for _ in range(cout)] for _ in range(1)]


def build_program() -> VcProgram:
    program = VcProgram("multi_layer_chain")

    n, h, w = 1, 32, 32
    cin0, cout0 = 16, 24
    cin1, cout1 = cout0, 24

    # Tensors
    ifm0 = VcTensor("ifm0", shape=(n, cin0, h, w), layout="NCHW", dtype="int8", data=_make_input(n, cin0, h, w))
    ofm0 = VcTensor("ofm0", shape=(n, cout0, h, w), layout="NCHW", dtype="int8")
    ifm1 = ofm0  # next layer input reuses ofm0 tensor name
    ofm1 = VcTensor("ofm1", shape=(n, cout1, h, w), layout="NCHW", dtype="int8")

    w0 = VcTensor("weight0", shape=(cout0, cin0, 3, 3), layout="NCHW", dtype="int8", data=_make_weight(cout0, cin0, 3, 3))
    b0 = VcTensor("bias0", shape=(1, cout0, 1, 1), layout="NCHW", dtype="int32", data=_make_bias(cout0))
    w1 = VcTensor("weight1", shape=(cout1, cin1, 1, 1), layout="NCHW", dtype="int8", data=_make_weight(cout1, cin1, 1, 1))
    b1 = VcTensor("bias1", shape=(1, cout1, 1, 1), layout="NCHW", dtype="int32", data=_make_bias(cout1))

    for t in (ifm0, ofm0, ofm1, w0, b0, w1, b1):
        program.add_tensor(t)

    conv0 = VcConv2D(
        name="conv0",
        inputs=["ifm0"],
        outputs=["ofm0"],
        weight="weight0",
        bias="bias0",
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        activation="relu",
    )
    pw1 = VcPointwiseConv(
        name="pw1",
        inputs=["ofm0"],
        outputs=["ofm1"],
        weight="weight1",
        bias="bias1",
        activation="relu",
    )

    program.add_op(conv0)
    program.add_op(pw1)
    return program


def main() -> None:
    ifm_base = 0x0000_0000
    ofm_base = None  # let ping-pong place B after A
    param_base = 0x1000_0000

    hw = HwConfig(
        ibuf_line_bytes=3840,
        wbuf_lane_bytes=2048,
        wbuf_lanes=4,
        max_h_tile=32,
        cluster_count=2,
    )

    program = build_program()
    out_dir = Path("out/examples/multi_layer_chain")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_ir(program, out_dir / "debug_ir.json")

    artifact = compile_program(
        program,
        output_dir=out_dir,
        target="venuscore-v1",
        dump_ir=True,
        dump_uop=True,
        ifm_base=ifm_base,
        ofm_base=ofm_base,
        param_base=param_base,
        hw=hw,
    )
    print(
        f"Generated {len(artifact.uops)} uOPs across {len(program.ops)} layers; "
        f"activation_peak={artifact.metadata.get('activation_peak_bytes','n/a')} bytes, "
        f"param_size={artifact.metadata.get('weight_bytes','n/a')} bytes, "
        f"final_ofm_base={artifact.metadata.get('output_base','n/a')}."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
