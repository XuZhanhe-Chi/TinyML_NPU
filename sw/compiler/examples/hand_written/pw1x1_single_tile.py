# -*- coding: utf-8 -*-
"""
Demonstrates compiling a handwritten single-tile pointwise (1x1) conv program.

Outputs are written to out/examples/pw1x1_single_tile/.
"""

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.ir.tensor import VcTensor
from venuscore_compiler.ir.ops import VcPointwiseConv
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


def _make_weight(cout: int, cin: int) -> list:
    """Create a small signed int8 kernel pattern shaped (Cout, Cin, 1, 1)."""
    weights = []
    for co in range(cout):
        cin_list = []
        for ci in range(cin):
            cin_list.append([[[(co + ci) % 4 - 2]]])  # shape (1,1), range [-2,1]
        weights.append(cin_list)
    return weights


def _make_bias(cout: int) -> list:
    """Create a bias tensor shaped (1, Cout, 1, 1) filled with zeros."""
    return [[[[0] for _ in range(1)] for _ in range(cout)] for _ in range(1)]


def build_pw1x1_program(
    hin: int = 64,
    win: int = 64,
    cin: int = 40,
    cout: int = 24,
) -> VcProgram:
    """Construct a simple pointwise conv program."""

    program = VcProgram("pw1x1_program")
    ifm = VcTensor(name="input", shape=(1, cin, hin, win), layout="NCHW", dtype="int8", data=_make_input(1, cin, hin, win))
    ofm = VcTensor(name="output", shape=(1, cout, hin, win), layout="NCHW", dtype="int8")
    weight = VcTensor(name="weight", shape=(cout, cin, 1, 1), layout="NCHW", dtype="int8", data=_make_weight(cout, cin))
    bias = VcTensor(name="bias", shape=(1, cout, 1, 1), layout="NCHW", dtype="int32", data=_make_bias(cout))

    program.add_tensor(ifm)
    program.add_tensor(ofm)
    program.add_tensor(weight)
    program.add_tensor(bias)

    op = VcPointwiseConv(
        name="pw0",
        inputs=["input"],
        outputs=["output"],
        weight="weight",
        bias="bias",
        activation="none",
    )
    program.add_op(op)
    return program


def main() -> None:
    program = build_pw1x1_program()
    out_dir = Path("out/examples/pw1x1_single_tile")
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
