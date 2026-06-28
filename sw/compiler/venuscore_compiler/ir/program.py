# -*- coding: utf-8 -*-
"""
Module overview:
  - Defines the VcProgram container that owns logical tensors and operations for VenusCore compilation.
  - Dependencies:
    * Depends on: venuscore_compiler.ir.ops, venuscore_compiler.ir.tensor.
    * Used by: frontend graph builders and backend compilation pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List

from venuscore_compiler.ir.ops import VcOp
from venuscore_compiler.ir.tensor import VcTensor


class VcProgram:
    """Holds tensors and ops for a VenusCore compilation."""

    def __init__(self, name: str = "venuscore_program") -> None:
        self.name = name
        self.tensors: Dict[str, VcTensor] = {}
        self.ops: List[VcOp] = []
        # Optional metadata hook for frontends/midend to attach auxiliary info.
        self.metadata: Dict[str, object] = {}

    def add_tensor(self, tensor: VcTensor) -> None:
        """Register a tensor with the program."""

        self.tensors[tensor.name] = tensor

    def add_op(self, op: VcOp) -> None:
        """Append an operation to the program."""

        self.ops.append(op)

    def validate(self) -> None:
        """Lightweight consistency checks."""

        # Ensure all tensor names referenced by ops are registered in the program.
        missing: List[str] = []
        for op in self.ops:
            referenced: List[str] = list(op.inputs) + list(op.outputs)
            if getattr(op, "weight", None):
                referenced.append(op.weight)  # type: ignore[arg-type]
            if getattr(op, "bias", None):
                referenced.append(op.bias)  # type: ignore[arg-type]
            for t_name in referenced:
                if t_name not in self.tensors and t_name not in missing:
                    missing.append(t_name)
        if missing:
            raise ValueError(f"Program '{self.name}' is missing tensors referenced by ops (inputs/outputs/weights/biases): {missing}")

    def to_dict(self) -> Dict[str, object]:
        """Serialize the program to a dictionary for debugging."""

        return {
            "name": self.name,
            # Avoid dumping large/raw tensor payloads (e.g., data blobs) to keep debug JSON manageable.
            "tensors": {
                name: (
                    t.to_dict()
                    if hasattr(t, "to_dict")
                    else {k: v for k, v in asdict(t).items() if k != "data"}
                )
                for name, t in self.tensors.items()
            },
            "ops": [op.to_dict() if hasattr(op, "to_dict") else asdict(op) for op in self.ops],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the program to a JSON string for inspection or logging."""

        return json.dumps(self.to_dict(), indent=indent)

    def topological_order(self) -> List[VcOp]:
        """Return ops in a simple insertion/topological order."""

        # TODO: currently uses insertion order; replace with a true topo-sort for branching/multi-input graphs.
        return list(self.ops)
