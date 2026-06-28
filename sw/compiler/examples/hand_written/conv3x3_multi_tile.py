# -*- coding: utf-8 -*-
"""
Multi-tile Conv3x3 example to exercise H tiling and ping-pong memory planning.
"""

from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.config import HwConfig
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
    # Geometry chosen to force multiple H stripes by limiting max_h_tile.
    n, cin, cout = 1, 32, 24
    hin, win = 64, 64
    padding = (1, 1, 1, 1)
    stride = (1, 1)

    # Ping-pong base addresses (adjust to your memory map).
    # ifm_base = 0x0000_0000
    # ofm_base = None
    # param_base = 0x1000_0000
    
    ifm_base = None
    ofm_base = None
    param_base = None
    
    
    # Force H tiling: small max_h_tile, two clusters for demonstration.
    hw = HwConfig(
        ibuf_line_bytes=3840,
        wbuf_lane_bytes=2048,
        wbuf_lanes=4,
        max_h_tile=16,
        cluster_count=2,
    )

    program = build_single_conv3x3_program(
        name="conv3x3_multi_tile",
        hin=hin,
        win=win,
        cin=cin,
        cout=cout,
        padding=padding,
        stride=stride,
        input_data=_make_input(n, cin, hin, win),
        weight_data=_make_weight(cout, cin, 3, 3),
        bias_data=_make_bias(cout),
        input_scale=1.0,
        weight_scale=[1.0 for _ in range(cout)],
        output_scale=1.0,
    )

    out_dir = Path("out/examples/conv3x3_multi_tile")
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
        f"Generated {len(artifact.uops)} uOPs; "
        f"activation_peak={artifact.metadata.get('activation_peak_bytes', 'n/a')} bytes, "
        f"param_size={artifact.metadata.get('weight_bytes', 'n/a')} bytes. "
        f"See {out_dir} for debug dumps."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
