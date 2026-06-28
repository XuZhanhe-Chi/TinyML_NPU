# -*- coding: utf-8 -*-
"""
Demonstrates compiling a handwritten single-layer FullyConnected program.

The midend normalizer lowers VcFullyConnected into a 1x1 VcPointwiseConv,
so the backend will generate PWCONV uOPs and a standard conv-style Param Block.

Outputs are written to out/examples/fc_single_tile/.
"""

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.ir.ops import VcFullyConnected
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.ir.tensor import VcTensor
from venuscore_compiler.utils.debug_dump import dump_ir


def _make_input(cin: int) -> list:
    """Create a small int8 input vector as a 4D NCHW tensor [1, Cin, 1, 1]."""
    return [[[[((ci * 3) % 13) - 6]] for ci in range(cin)]]


def _make_weight_2d(cout: int, cin: int) -> list:
    """
    Create a nested 2D int8 weight matrix shaped [Cout][Cin].

    The midend FC->PW lowering will flatten this into [Cout, Cin, 1, 1].
    """
    w: list[list[int]] = []
    for co in range(cout):
        row = []
        for ci in range(cin):
            row.append(((co + ci * 7) % 19) - 9)  # range [-9, 9]
        w.append(row)
    return w


def _make_bias(cout: int) -> list:
    """Create an int32 bias vector shaped [Cout]."""
    return [0 for _ in range(cout)]


def build_fc_program(cin: int = 8, cout: int = 12) -> VcProgram:
    """Build a single FullyConnected op IR program (H=W=1)."""
    program = VcProgram("fc_single_tile")

    ifm = VcTensor(name="input", shape=(1, cin, 1, 1), layout="NCHW", dtype="int8", data=_make_input(cin))
    ofm = VcTensor(name="output", shape=(1, cout, 1, 1), layout="NCHW", dtype="int8")

    # Use 4D shape (Cout, Cin, 1, 1) but store data as nested 2D [Cout][Cin] to
    # exercise the FC->PW weight normalization path.
    weight = VcTensor(
        name="weight",
        shape=(cout, cin, 1, 1),
        layout="NCHW",
        dtype="int8",
        data=_make_weight_2d(cout, cin),
    )
    bias = VcTensor(name="bias", shape=(cout, 1, 1, 1), layout="NCHW", dtype="int32", data=_make_bias(cout))

    program.add_tensor(ifm)
    program.add_tensor(ofm)
    program.add_tensor(weight)
    program.add_tensor(bias)

    program.add_op(
        VcFullyConnected(
            name="fc0",
            inputs=["input"],
            outputs=["output"],
            weight="weight",
            bias="bias",
            activation=None,
            qmode=None,
        )
    )

    program.validate()
    return program


def main() -> None:
    out_dir = Path("out/examples/fc_single_tile")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional base addresses for IFM/OFM/Param regions; adjust to match your memory map.
    ifm_base = 0x0000_0000
    ofm_base = 0x1000_0000
    param_base = 0x2000_0000

    program = build_fc_program(cin=8, cout=12)
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
    )

    print(
        f"Generated {len(artifact.uops)} uOPs; "
        f"activation_peak={artifact.metadata.get('activation_peak_bytes','n/a')} bytes, "
        f"param_size={artifact.metadata.get('weight_bytes','n/a')} bytes, "
        f"final_ofm_base={artifact.metadata.get('output_base','n/a')}."
    )


if __name__ == "__main__":  # pragma: no cover
    main()

