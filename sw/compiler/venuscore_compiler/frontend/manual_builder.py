# -*- coding: utf-8 -*-
"""
Module overview:
  - Provides helpers to build hand-written IR programs for VenusCore.
  - Dependencies:
    * Depends on: venuscore_compiler.ir.program, venuscore_compiler.ir.ops, venuscore_compiler.ir.tensor.
    * Used by: hand-written examples and CLI smoke fallback.
"""

from __future__ import annotations

from collections.abc import Sequence

from venuscore_compiler.ir.ops import VcConv2D
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.ir.tensor import VcTensor


def build_single_conv3x3_program(
    name: str = "conv3x3_single",
    hin: int = 28,
    win: int = 28,
    cin: int = 8,
    cout: int = 8,
    padding: tuple[int, int, int, int] = (1, 1, 1, 1),
    stride: tuple[int, int] = (1, 1),
    input_data: object | None = None,
    weight_data: object | None = None,
    bias_data: object | None = None,
    input_scale: float | None = None,
    weight_scale: float | Sequence[float] | None = None,
    output_scale: float | None = None,
) -> VcProgram:
    """
    Create a simple single-layer Conv3x3 program with configurable geometry and optional data payloads.

    If no data is provided, tensors are initialized with simple zero-filled lists to allow downstream
    param-block packing without None checks.
    """

    program = VcProgram(name=name)

    def _zeros_4d(n: int, c: int, h: int, w: int, value: int = 0) -> list[list[list[list[int]]]]:
        # Produce a simple 4D nested list filled with a constant value.
        return [[[[value for _ in range(w)] for _ in range(h)] for _ in range(c)] for _ in range(n)]

    input_tensor = VcTensor(
        name="input",
        shape=(1, cin, hin, win),
        dtype="int8",
        data=input_data if input_data is not None else _zeros_4d(1, cin, hin, win, value=1),
        scale=input_scale,
        q_scheme="symmetric_per_tensor" if input_scale is not None else "none",
    )
    weight_tensor = VcTensor(
        name="weight",
        shape=(cout, cin, 3, 3),
        dtype="int8",
        data=weight_data if weight_data is not None else _zeros_4d(cout, cin, 3, 3, value=1),
        scale=weight_scale,
        q_scheme="symmetric_per_channel"
        if isinstance(weight_scale, Sequence) and not isinstance(weight_scale, (str, bytes))
        else ("symmetric_per_tensor" if weight_scale is not None else "none"),
        q_axis=0 if isinstance(weight_scale, Sequence) else None,
    )
    bias_tensor = VcTensor(
        name="bias",
        shape=(1, cout, 1, 1),
        dtype="int32",
        data=bias_data if bias_data is not None else _zeros_4d(1, cout, 1, 1, value=0),
    )
    output_tensor = VcTensor(
        name="output",
        shape=(1, cout, 0, 0),  # placeholder, updated below
        dtype="int8",
        scale=output_scale,
        q_scheme="symmetric_per_tensor" if output_scale is not None else "none",
    )

    pad_top, pad_bottom, pad_left, pad_right = padding
    stride_h, stride_w = stride
    # Output shape for stride 1/2 with padding applied (floor division by design).
    hout = (hin + pad_top + pad_bottom - 3) // stride_h + 1
    wout = (win + pad_left + pad_right - 3) // stride_w + 1
    output_tensor.shape = (1, cout, hout, wout)

    program.add_tensor(input_tensor)
    program.add_tensor(weight_tensor)
    program.add_tensor(bias_tensor)
    program.add_tensor(output_tensor)

    conv = VcConv2D(
        name="conv0",
        inputs=["input"],
        outputs=["output"],
        weight="weight",
        bias="bias",
        kernel=(3, 3),
        stride=stride,
        padding=padding,
        activation="relu",
        qmode=None,
    )
    program.add_op(conv)
    program.validate()
    return program
