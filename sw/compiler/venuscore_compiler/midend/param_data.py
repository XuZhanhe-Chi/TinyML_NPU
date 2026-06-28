# -*- coding: utf-8 -*-
"""
Param data extraction: produce a lightweight view of weight/bias tensors for backend consumption.

Returns a mapping: name -> {"shape": tuple, "dtype": str, "data": tensor.data}
"""

from __future__ import annotations

from typing import Any, Dict

from venuscore_compiler.ir.program import VcProgram

ParamData = Dict[str, Dict[str, Any]]

__all__ = ["ParamData", "build_param_data"]


def build_param_data(program: VcProgram) -> ParamData:
    """Extract shapes/dtypes/data for all tensors; backend can consume without importing IR types."""

    view: ParamData = {}
    for name, tensor in program.tensors.items():
        view[name] = {
            "shape": getattr(tensor, "shape", None),
            "dtype": getattr(tensor, "dtype", "int8"),
            "data": getattr(tensor, "data", None),
        }
    return view


def prepare_param_buffers(param_data: ParamData, tile_plan=None) -> ParamData:
    """
    Placeholder for pre-packing weights/bias into hardware layout.

    Currently returns param_data unchanged; future revisions can pack weights per tile/op.
    """

    return param_data
