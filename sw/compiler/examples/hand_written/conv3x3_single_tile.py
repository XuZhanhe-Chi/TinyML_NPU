# -*- coding: utf-8 -*-
"""
Module overview:
  - Demonstrates compiling a handwritten single-tile conv3x3 program.
  - Dependencies:
    * Depends on: venuscore_compiler.frontend.manual_builder, venuscore_compiler.compile_program.
    * Used by: users wanting a quick end-to-end sanity check.
"""

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.frontend.manual_builder import build_single_conv3x3_program
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


def _make_weight(cout: int, cin: int, kh: int, kw: int) -> list:
    """Create a small signed int8 kernel pattern shaped (Cout, Cin, Kh, Kw)."""

    weights = []
    for co in range(cout):
        cin_list = []
        for ci in range(cin):
            kernel = []
            for y in range(kh):
                row = []
                for x in range(kw):
                    row.append(((co + ci + x + y) % 5) - 2)  # range [-2, 2]
                kernel.append(row)
            cin_list.append(kernel)
        weights.append(cin_list)
    return weights


def _make_bias(cout: int) -> list:
    """Create a bias tensor shaped (1, Cout, 1, 1) filled with zeros."""

    return [[[[0] for _ in range(1)] for _ in range(cout)] for _ in range(1)]


def main() -> None:
    # Configure geometry once and reuse for both data construction and builder parameters.
    n, cin, cout = 1, 24, 24
    
    hin, win = 8, 8
    padding = (1, 1, 1, 1)
    stride = (1, 1)
    # Optional base addresses for IFM/OFM/Param regions; adjust to match your memory map.
    ifm_base = 0x0000_0000
    ofm_base = 0x1000_0000
    param_base = 0x2000_0000

    program = build_single_conv3x3_program(
        hin=hin,
        win=win,
        cin=cin,
        cout=cout,
        padding=padding,
        stride=stride,
        input_data=_make_input(n, cin, hin, win),
        weight_data=_make_weight(cout, cin, 3, 3),
        bias_data=_make_bias(cout),
        input_scale=1,  # example per-tensor scale
        weight_scale=[1.0 for _ in range(cout)],  # example per-channel scales
        output_scale=1,
    )
    dump_ir(program, Path("out/examples/conv3x3_single_tile/debug_ir.json"))
    artifact = compile_program(
        program,
        output_dir=Path("out/examples/conv3x3_single_tile"),
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
