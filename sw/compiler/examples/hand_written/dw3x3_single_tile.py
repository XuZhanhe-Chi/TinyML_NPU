# -*- coding: utf-8 -*-
"""
Demonstrates compiling a handwritten single-tile depthwise 3x3 program.

Outputs are written to out/examples/dw3x3_single_tile/.
"""

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.ir.tensor import VcTensor
from venuscore_compiler.ir.ops import VcDepthwiseConv
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.utils.debug_dump import dump_ir


def _make_input(n: int, c: int, h: int, w: int) -> list:
    """Create an int8-friendly NCHW input pattern."""
    data = []
    for _ in range(n):
        channels = []
        for _ in range(c):
            plane = []
            for y in range(h):
                row = []
                for x in range(w):
                    row.append((x + y) % 8)  # small positive pattern
                plane.append(row)
            channels.append(plane)
        data.append(channels)
    return data


def _make_weight(cin: int) -> list:
    """Create a small signed int8 DW kernel pattern shaped (Cin, 1, 3, 3) (ONNX-style)."""
    weights = []
    for ch in range(cin):
        kernel = []
        for y in range(3):
            row = []
            for x in range(3):
                row.append(((ch + x + y) % 5) - 2)  # range [-2, 2]
            kernel.append(row)
        # [Cin, 1, 3, 3]
        weights.append([kernel])
    return weights


def _make_bias(cin: int) -> list:
    """Create a bias tensor shaped (1, Cin, 1, 1) filled with zeros."""
    return [[[[0] for _ in range(1)] for _ in range(cin)] for _ in range(1)]


def build_dw3x3_program(
    hin: int = 8,
    win: int = 8,
    cin: int = 8,
) -> VcProgram:
    """Construct a simple depthwise conv program."""

    program = VcProgram("dw3x3_program")
    ifm = VcTensor(name="input", shape=(1, cin, hin, win), layout="NCHW", dtype="int8", data=_make_input(1, cin, hin, win))
    ofm = VcTensor(name="output", shape=(1, cin, hin, win), layout="NCHW", dtype="int8")
    weight = VcTensor(name="weight", shape=(cin, 1, 3, 3), layout="NCHW", dtype="int8", data=_make_weight(cin))
    bias = VcTensor(name="bias", shape=(1, cin, 1, 1), layout="NCHW", dtype="int32", data=_make_bias(cin))

    program.add_tensor(ifm)
    program.add_tensor(ofm)
    program.add_tensor(weight)
    program.add_tensor(bias)

    op = VcDepthwiseConv(
        name="dw0",
        inputs=["input"],
        outputs=["output"],
        weight="weight",
        bias="bias",
        activation="none",
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        groups=cin,
    )
    program.add_op(op)
    return program


def main() -> None:
    program = build_dw3x3_program()
    out_dir = Path("out/examples/dw3x3_single_tile")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_ir(program, out_dir / "debug_ir.json")
    artifact = compile_program(
        program,
        output_dir=out_dir,
        target="venuscore-v1",
        dump_ir=True,
        dump_uop=True,
    )
    print(f"Generated {len(artifact.uops)} uOPs; metadata written to {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
