"""
Tests for FIRST/LAST/SYNC flag semantics in backend uOP generation.
"""

from __future__ import annotations

from venuscore_compiler.backend.codegen_uop import generate_uops
from venuscore_compiler.backend.memory_planner import MemoryPlan
from venuscore_compiler.midend.types import LayerConfig, TileDesc, TilePlan


def _mk_layer_cfg(layer_id: int, *, op_type: str, cin: int, cout: int, ifm_hw: tuple[int, int], ofm_hw: tuple[int, int]) -> LayerConfig:
    ifm_h, ifm_w = ifm_hw
    ofm_h, ofm_w = ofm_hw
    return LayerConfig(
        layer_id=layer_id,
        name=f"op{layer_id}",
        op_type=op_type,
        act_type=None,
        ifm_h=ifm_h,
        ifm_w=ifm_w,
        ofm_h=ofm_h,
        ofm_w=ofm_w,
        cin=cin,
        cout=cout,
        c4_in=(cin + 3) // 4,
        c4_out=(cout + 3) // 4,
        stride_h=1,
        stride_w=1,
        pad_top=0,
        pad_bottom=0,
        pad_left=0,
        pad_right=0,
        fi_stride=ifm_h * ifm_w,
        fo_stride=ofm_h * ofm_w,
        ifm_name="ifm",
        ofm_name="ofm",
        input_name="ifm",
        output_name="ofm",
        weight_name="w",
        bias_name="b",
        qmode="INT8",
    )


def test_first_last_are_per_output_tile_group_not_per_layer() -> None:
    """
    With Cout tiling (multiple tiles per layer), the current compiler does not
    Cin-tile. FIRST/LAST are stream-level markers. Therefore:
      - FIRST_FLAG == 1 only for the first uOP in the stream
      - LAST_FLAG  == 1 only for the last uOP in the stream
      - SYNC == 1 only for the last tile of the layer
    """

    layer = _mk_layer_cfg(0, op_type="pointwise_conv", cin=8, cout=128, ifm_hw=(8, 8), ofm_hw=(8, 8))

    tiles = []
    stripes = [(0, 44), (44, 88), (88, 128)]
    for tid, (co0, co1) in enumerate(stripes):
        tiles.append(
            TileDesc(
                layer_id=0,
                op_name="op0",
                op_type="pointwise_conv",
                h_tile=8,
                w_tile=8,
                y_index=0,
                pad_top=0,
                pad_bottom=0,
                pad_left=0,
                pad_right=0,
                cin=8,
                cout=co1 - co0,
                co_start=co0,
                co_end=co1,
                c4_in_start=0,
                c4_in_end=(8 + 3) // 4,
                c4_out_start=co0 // 4,
                c4_out_end=(co1 + 3) // 4,
                kernel_h=1,
                kernel_w=1,
                stride_h=1,
                stride_w=1,
                h_in=8,
                w_in=8,
                input_name="ifm",
                output_name="ofm",
                weight_name="w",
                bias_name="b",
                tile_id=tid,
                tile_index=tid,
                is_first_in_layer=(tid == 0),
                is_last_in_layer=(tid == (len(stripes) - 1)),
            )
        )

    tp = TilePlan(tiles=tiles, tiles_by_op={"op0": list(tiles)}, layers=[layer])
    mp = MemoryPlan(
        ifm_offsets={"ifm": 0},
        ofm_offsets={"ofm": 0},
        param_base=0,
        param_offsets={("op0", 0): 0, ("op0", 1): 0x1000, ("op0", 2): 0x2000},
        layer_configs={0: layer},
    )

    uops = generate_uops(tp, mp)
    assert len(uops) == 3

    assert [u.first_flag for u in uops] == [True, False, False]
    assert [u.last_flag for u in uops] == [False, False, True]
    assert [u.sync for u in uops] == [False, False, True]


def test_sync_is_layer_barrier_even_when_not_last_of_stream() -> None:
    """
    SYNC is a layer-level barrier: it should be asserted on the last tile of
    each layer, even if the full stream contains multiple layers.
    """

    l0 = _mk_layer_cfg(0, op_type="conv2d", cin=4, cout=4, ifm_hw=(4, 4), ofm_hw=(4, 4))
    l1 = _mk_layer_cfg(1, op_type="avg_pool", cin=4, cout=4, ifm_hw=(4, 4), ofm_hw=(2, 2))

    t0 = TileDesc(
        layer_id=0,
        op_name="op0",
        op_type="conv2d",
        h_tile=4,
        w_tile=4,
        y_index=0,
        pad_top=1,
        pad_bottom=1,
        pad_left=1,
        pad_right=1,
        cin=4,
        cout=4,
        co_start=0,
        co_end=4,
        c4_in_start=0,
        c4_in_end=1,
        c4_out_start=0,
        c4_out_end=1,
        kernel_h=3,
        kernel_w=3,
        stride_h=1,
        stride_w=1,
        h_in=4,
        w_in=4,
        input_name="ifm",
        output_name="mid",
        weight_name="w0",
        bias_name="b0",
        tile_id=0,
        tile_index=0,
        is_first_in_layer=True,
        is_last_in_layer=True,
    )
    t1 = TileDesc(
        layer_id=1,
        op_name="op1",
        op_type="avg_pool",
        h_tile=2,
        w_tile=2,
        y_index=0,
        pad_top=0,
        pad_bottom=0,
        pad_left=0,
        pad_right=0,
        cin=4,
        cout=4,
        co_start=0,
        co_end=4,
        c4_in_start=0,
        c4_in_end=1,
        c4_out_start=0,
        c4_out_end=1,
        kernel_h=2,
        kernel_w=2,
        stride_h=2,
        stride_w=2,
        h_in=4,
        w_in=4,
        input_name="mid",
        output_name="ofm",
        weight_name="",
        bias_name="",
        tile_id=0,
        tile_index=1,
        is_first_in_layer=True,
        is_last_in_layer=True,
    )

    tp = TilePlan(tiles=[t0, t1], tiles_by_op={"op0": [t0], "op1": [t1]}, layers=[l0, l1])
    mp = MemoryPlan(
        ifm_offsets={"ifm": 0, "mid": 0x1000},
        ofm_offsets={"mid": 0x1000, "ofm": 0x2000},
        param_base=0,
        param_offsets={("op0", 0): 0, ("op1", 0): 0x1000},
        layer_configs={0: l0, 1: l1},
    )

    uops = generate_uops(tp, mp)
    assert len(uops) == 2
    assert [u.first_flag for u in uops] == [True, False]
    assert [u.last_flag for u in uops] == [False, True]
    assert [u.sync for u in uops] == [True, True]
