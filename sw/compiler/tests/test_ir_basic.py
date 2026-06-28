# -*- coding: utf-8 -*-
"""
Module overview:
  - Basic unit tests for IR tensor/op/program construction.
  - Dependencies:
    * Depends on: venuscore_compiler.ir.tensor, venuscore_compiler.ir.ops, venuscore_compiler.ir.program
    * Used by: CI/developers to validate IR construction
"""

from venuscore_compiler.ir.ops import VcConv2D
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.ir.tensor import VcTensor


def test_program_construction():
    program = VcProgram()
    input_tensor = VcTensor(name="input", shape=(1, 3, 8, 8))
    weight_tensor = VcTensor(name="weight", shape=(8, 3, 3, 3))
    bias_tensor = VcTensor(name="bias", shape=(1, 8, 1, 1))
    output_tensor = VcTensor(name="output", shape=(1, 8, 6, 6))

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
    )
    program.add_op(conv)
    program.validate()

    assert "input" in program.tensors
    assert program.ops[0].op_type == "conv2d"
