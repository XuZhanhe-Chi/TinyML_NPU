# -*- coding: utf-8 -*-
"""
IR passes for normalization and legalization.

These passes operate purely on the logical IR (VcProgram/VcOp/VcTensor) and do not
introduce hardware-specific fields (addresses, tiling, etc.). The intent is to keep
frontend output clean and only perform light canonicalization before midend/backend.
"""

from __future__ import annotations

from typing import Callable, Iterable

from venuscore_compiler.ir.program import VcProgram

__all__ = [
    "run_passes",
    "normalize_layouts",
    "legalize_program",
    "DEFAULT_PASSES",
]

# Default pipeline applied by run_passes when no custom list is provided.
DEFAULT_PASSES: tuple[Callable[[VcProgram], VcProgram], ...] = (
    normalize_layouts,
    legalize_program,
)


def run_passes(program: VcProgram, passes: Iterable[Callable[[VcProgram], VcProgram]] | None = None) -> VcProgram:
    """
    Execute a sequence of IR passes in order, returning the transformed program.

    The input program is mutated in-place by each pass and also returned for convenience.
    """

    pipeline = passes if passes is not None else DEFAULT_PASSES
    current = program
    for p in pipeline:
        current = p(current)
    return current


def normalize_layouts(program: VcProgram) -> VcProgram:
    """
    Normalize tensor layout metadata to a canonical form.

    Current behavior:
      - Ensure tensor.layout is uppercase (e.g., "NCHW") for consistency.
    This is a lightweight placeholder; more complex layout conversions should be added
    when multiple layouts are supported.
    """

    for tensor in program.tensors.values():
        if tensor.layout:
            tensor.layout = tensor.layout.upper()
    return program


def legalize_program(program: VcProgram) -> VcProgram:
    """
    Apply lightweight legalization to make the IR friendlier to backend constraints.

    Current behavior:
      - No structural changes; serves as a hook for future passes such as channel padding
        to multiples of 4 or dtype canonicalization.
    """

    # TODO: pad channels to hardware-friendly multiples (e.g., 4) when backend requires it.
    # TODO: insert explicit bias tensors if ops reference missing bias names.
    return program
