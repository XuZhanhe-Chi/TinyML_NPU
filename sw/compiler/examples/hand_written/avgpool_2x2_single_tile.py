# -*- coding: utf-8 -*-
"""
Demonstrates compiling a handwritten single-tile AvgPool 2x2 program.

Outputs are written to out/examples/avgpool_2x2_single_tile/.
"""

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.ir.tensor import VcTensor
from venuscore_compiler.ir.ops import VcAvgPool
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


def build_avgpool_program(
    hin: int = 4,
    win: int = 4,
    cin: int = 4,
) -> VcProgram:
    """Construct a simple avgpool 2x2 program."""

    program = VcProgram("avgpool_2x2_program")
    ifm = VcTensor(name="input", shape=(1, cin, hin, win), layout="NCHW", dtype="int8", data=_make_input(1, cin, hin, win))
    # 2x2 kernel, stride 2 => H/2, W/2
    ofm = VcTensor(name="output", shape=(1, cin, hin // 2, win // 2), layout="NCHW", dtype="int8")

    program.add_tensor(ifm)
    program.add_tensor(ofm)

    op = VcAvgPool(
        name="pool0",
        inputs=["input"],
        outputs=["output"],
        kernel=(2, 2),
        stride=(2, 2),
        padding=(0, 0, 0, 0),
    )
    program.add_op(op)
    return program


def main() -> None:
    program = build_avgpool_program()
    out_dir = Path("out/examples/avgpool_2x2_single_tile")
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
