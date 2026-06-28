# -*- coding: utf-8 -*-
"""
Module overview:
  - Utility functions public API (aggregated imports).
  - Dependencies:
    * Depends on: utils.math_utils, utils.debug_dump, utils.model_analysis
    * Used by: backend and scripts
"""


from venuscore_compiler.utils.debug_dump import dump_program_to_json, dump_uops
from venuscore_compiler.utils.math_utils import compute_output_dim, pad_to_multiple
from venuscore_compiler.utils.model_analysis import estimate_macs, estimate_model_size

__all__ = [
    "compute_output_dim",
    "dump_program_to_json",
    "dump_uops",
    "estimate_macs",
    "estimate_model_size",
    "pad_to_multiple",
]
