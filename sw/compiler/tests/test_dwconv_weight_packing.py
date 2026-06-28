# -*- coding: utf-8 -*-
"""
Regression tests for DWCONV weight packing against sw/compiler/doc/VenusCore_MemortMap.md (4.4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from venuscore_compiler import compile_program
from venuscore_compiler.config import HwConfig
from venuscore_compiler.ir.ops import VcDepthwiseConv
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.ir.tensor import VcTensor


def _s8(x: int) -> int:
    return x - 256 if x >= 128 else x


def _u8(x: int) -> int:
    return x & 0xFF


def _pack_dw_word(top: int, mid: int, bot: int) -> bytes:
    # little-endian u32: byte0=top, byte1=mid, byte2=bot, byte3=0
    return bytes([_u8(top), _u8(mid), _u8(bot), 0])


def test_dwconv_weight_block_matches_spec_with_channel_tiling(tmp_path: Path) -> None:
    """
    Force Cout tiling on DWCONV (Cin==Cout) and verify each tile's Weight Data Block
    follows:

      for g in 0..C4_dw-1:
        for lane in 0..3:
          ch = 4*g + lane
          for kw in 0..2:
            emit {K[ch][0][kw], K[ch][1][kw], K[ch][2][kw], 0}
    """

    cin = cout = 8
    hin = win = 1
    padding = (1, 1, 1, 1)
    stride = (1, 1)

    program = VcProgram(name="dw_pack")
    program.add_tensor(VcTensor(name="input", shape=(1, cin, hin, win), dtype="int8", data=[[[[1]] for _ in range(cin)]]))

    # ONNX-style DW weight layout: [Cin, 1, 3, 3]
    w = np.zeros((cin, 1, 3, 3), dtype=np.int8)
    for ch in range(cin):
        for kh in range(3):
            for kw in range(3):
                w[ch, 0, kh, kw] = np.int8(((ch * 11 + kh * 3 + kw) % 127) - 63)
    program.add_tensor(VcTensor(name="weight", shape=(cin, 1, 3, 3), dtype="int8", data=w.tolist()))
    program.add_tensor(VcTensor(name="bias", shape=(1, cout, 1, 1), dtype="int32", data=[[[[0]] for _ in range(cout)]]))
    program.add_tensor(VcTensor(name="output", shape=(1, cout, 1, 1), dtype="int8"))

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
            activation=None,
            qmode=None,
        )
    )
    program.validate()

    # Force channel tiling via cluster balancing: with H stripes=1 and cluster_count=4,
    # Cout=8 will be split into 2 stripes of 4 channels.
    hw = HwConfig(cluster_count=4, wbuf_lane_bytes=(1 << 16), wbuf_lanes=4, ibuf_line_bytes=3840, max_h_tile=255)
    artifact = compile_program(program, output_dir=tmp_path, hw=hw, dump_ir=False, dump_uop=True)

    # Expect 2 tiles (co: [0,4), [4,8)).
    assert len(artifact.uops) == 2

    params = (tmp_path / "params.bin").read_bytes()

    # Parse per-tile blocks:
    #   Quant block = Cout_tile*8 = 4*8 = 32 bytes
    #   align to 16 => already 32
    #   Weight bytes = Cin_tile * 3 * 4 = 4 * 12 = 48 bytes
    tile_block_bytes = 32 + 48
    assert len(params) == tile_block_bytes * 2

    for tile_idx, co_start in enumerate((0, 4)):
        base = tile_idx * tile_block_bytes
        weight_base = base + 32
        got = params[weight_base : weight_base + 48]

        exp = bytearray()
        # local Cin_dw = 4 channels in this tile
        for g in range(1):  # C4_dw=1
            for lane in range(4):
                ch_global = co_start + 4 * g + lane
                for kw in range(3):
                    top = int(w[ch_global, 0, 0, kw])
                    mid = int(w[ch_global, 0, 1, kw])
                    bot = int(w[ch_global, 0, 2, kw])
                    exp.extend(_pack_dw_word(top, mid, bot))

        assert got == bytes(exp)
