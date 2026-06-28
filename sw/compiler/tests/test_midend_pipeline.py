# -*- coding: utf-8 -*-
"""
Midend pipeline tests: normalization -> constraints -> layout lowering -> tiling -> quantize.
"""

from __future__ import annotations

import pytest

from venuscore_compiler.config import HwConfig
from venuscore_compiler.frontend.manual_builder import build_single_conv3x3_program
from venuscore_compiler.ir.ops import VcFullyConnected
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.ir.tensor import VcTensor
from venuscore_compiler.midend import run_midend
from venuscore_compiler.backend import memory_planner, layout_param_block, codegen_uop


def test_run_midend_produces_tileplan_and_quant() -> None:
    """Smoke test run_midend end-to-end on a small conv3x3 program."""

    program = build_single_conv3x3_program(hin=8, win=8, cin=8, cout=8, padding=(1, 1, 1, 1), stride=(1, 1))
    tile_plan, quant_table, qmode_map, param_data = run_midend(program)

    # Layer and tile presence
    assert tile_plan.layers, "Expected at least one LayerConfig"
    assert tile_plan.tiles, "Expected at least one TileDesc"

    layer = tile_plan.layers[0]
    tile = tile_plan.tiles[0]

    # Geometry and layout consistency
    assert layer.ifm_h == 8 and layer.ifm_w == 8
    assert layer.ofm_h == 8 and layer.ofm_w == 8
    assert layer.c4_in == (layer.cin + 3) // 4
    assert tile.w_tile == layer.ofm_w  # no W tiling
    assert tile.cin == layer.cin

    # Quant info propagated
    assert layer.name in quant_table
    assert layer.name in qmode_map
    assert len(layer.quant_table) == layer.cout

    # Param data contains weight/bias tensors
    assert "weight" in param_data and param_data["weight"]["shape"]


def test_run_midend_respects_hw_capacity() -> None:
    """IBUF line capacity violation should raise during tiling."""

    tiny_hw = HwConfig(
        ibuf_line_bytes=4,  # too small for cin=8, w=8
        wbuf_lane_bytes=1 << 10,
        wbuf_lanes=1,
        max_h_tile=64,
        cluster_count=1,
    )
    program = build_single_conv3x3_program(hin=8, win=8, cin=8, cout=8, padding=(1, 1, 1, 1), stride=(1, 1))

    with pytest.raises(ValueError):
        run_midend(program, hw=tiny_hw)


def test_backend_consumes_midend_outputs() -> None:
    """Backend planners/codegen should accept TilePlan + param data without IR."""

    program = build_single_conv3x3_program(hin=8, win=8, cin=8, cout=8, padding=(1, 1, 1, 1), stride=(1, 1))
    tile_plan, quant_table, _, param_data = run_midend(program)

    mp = memory_planner.plan_memory(tile_plan)
    uops = codegen_uop.generate_uops(tile_plan, mp)

    # Param block build should succeed with provided quant table and param data.
    blob = layout_param_block.build_param_block(tile_plan, mp, param_data, quant_table)
    assert blob and len(blob) > 0
    assert uops and len(uops) == len(tile_plan.tiles)


def test_fully_connected_is_lowered_to_pointwise() -> None:
    """FullyConnected should be lowered to a 1x1 pointwise conv and compiled."""

    cin, cout = 8, 12
    program = VcProgram("fc_to_pw")
    program.add_tensor(VcTensor(name="input", shape=(1, cin, 1, 1), dtype="int8", layout="NCHW"))
    program.add_tensor(VcTensor(name="output", shape=(1, cout, 1, 1), dtype="int8", layout="NCHW"))

    # Simulate ONNX Gemm weight as a 2D nested list [Cout][Cin], but keep 4D shape after forcing.
    weight_data = [[(co * 17 + ci) % 127 for ci in range(cin)] for co in range(cout)]
    program.add_tensor(VcTensor(name="weight", shape=(cout, cin, 1, 1), dtype="int8", layout="NCHW", data=weight_data))
    program.add_tensor(VcTensor(name="bias", shape=(cout, 1, 1, 1), dtype="int32", layout="NCHW", data=[0] * cout))

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

    tile_plan, quant_table, _, param_data = run_midend(program)
    assert tile_plan.layers and tile_plan.layers[0].op_type == "pointwise_conv"

    mp = memory_planner.plan_memory(tile_plan)
    uops = codegen_uop.generate_uops(tile_plan, mp)
    blob = layout_param_block.build_param_block(tile_plan, mp, param_data, quant_table)
    assert uops and blob
