# -*- coding: utf-8 -*-
"""
Module overview:
  - Graph traversal and topological utilities.
  - Dependencies:
    * Depends on: venuscore_compiler.ir.ops
    * Used by: frontend conversion and optimization passes
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set

from venuscore_compiler.ir.ops import VcOp


def topological_sort(ops: Iterable[VcOp]) -> List[VcOp]:
    """Perform a simple topological sort based on produced tensor names."""

    produced: Dict[str, VcOp] = {}
    for op in ops:
        for output in op.outputs:
            produced[output] = op

    visited: Set[VcOp] = set()
    ordered: List[VcOp] = []

    def visit(node: VcOp) -> None:
        if node in visited:
            return
        # Depth-first walk through producer relationships to honor data dependencies.
        for inp in node.inputs:
            parent = produced.get(inp)
            if parent:
                visit(parent)
        visited.add(node)
        ordered.append(node)

    for op in ops:
        visit(op)
    return ordered
