# -*- coding: utf-8 -*-
"""Frontend entrypoints retained for the public TinyML_NPU release."""

from venuscore_compiler.frontend.manual_builder import build_single_conv3x3_program
from venuscore_compiler.frontend.onnx_loader import load_onnx_to_ir

__all__ = [
    "build_single_conv3x3_program",
    "load_onnx_to_ir",
]
