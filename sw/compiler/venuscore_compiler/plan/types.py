# -*- coding: utf-8 -*-
"""
Plan data structures for NPU/CPU mixed execution.

These types are a compiler-side representation of the runtime ABI. The backend
can serialize them to JSON (metadata) and emit them as C structs/arrays
(`bundle.h`) for firmware to execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List


U16_MAX = 0xFFFF


class StepType(IntEnum):
    """Execution step type."""

    NPU = 0
    CPU = 1
    ALIAS = 2


class CpuKernel(IntEnum):
    """Built-in CPU fallback kernels (v1)."""

    ADD = 0
    CONCAT_C = 1


class CpuActivation(IntEnum):
    """Post-CPU activation kind for fallback kernels."""

    NONE = 0
    RELU = 1
    RELU6 = 2


@dataclass(frozen=True)
class TensorDesc:
    """
    Runtime tensor descriptor (activation arena placement).

    The physical layout is assumed to be NCHWc4 int8 (4 channels packed into a
    32-bit word per spatial element). `offset_bytes` is relative to an arena
    base pointer chosen by firmware.
    """

    tensor_id: int
    name: str
    offset_bytes: int
    size_bytes: int
    shape: tuple[int, int, int, int]  # logical NCHW
    layout: str = "NCHWc4"
    dtype: str = "int8"
    quant_index: int | None = None

    def __post_init__(self) -> None:
        if not (0 <= self.tensor_id <= U16_MAX):
            raise ValueError(f"tensor_id out of uint16 range: {self.tensor_id}")
        if self.offset_bytes < 0 or self.size_bytes < 0:
            raise ValueError("offset_bytes and size_bytes must be non-negative")
        if len(self.shape) != 4:
            raise ValueError(f"shape must be 4D NCHW, got {self.shape!r}")
        if self.layout != "NCHWc4":
            raise ValueError(f"Only NCHWc4 is supported in the plan v1, got layout={self.layout!r}")
        if self.dtype != "int8":
            raise ValueError(f"Only int8 activations are supported in the plan v1, got dtype={self.dtype!r}")
        if self.quant_index is not None and not (0 <= self.quant_index <= U16_MAX):
            raise ValueError(f"quant_index out of uint16 range: {self.quant_index}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "tensor_id": self.tensor_id,
            "name": self.name,
            "offset_bytes": self.offset_bytes,
            "size_bytes": self.size_bytes,
            "shape": list(self.shape),
            "layout": self.layout,
            "dtype": self.dtype,
            "quant_index": self.quant_index,
        }


@dataclass(frozen=True)
class StepDesc:
    """
    Base step descriptor.

    `inputs` and `outputs` are tensor IDs referencing entries in Plan.tensors.
    """

    # step_type is defined by the concrete subclass (NPU/CPU/ALIAS).
    # It is intentionally not part of the public constructor to avoid
    # call-site errors where the wrong type is passed.
    step_type: StepType = field(init=False, default=StepType.NPU)
    inputs: tuple[int, ...] = ()
    outputs: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.inputs) > 4:
            raise ValueError(f"Too many step inputs ({len(self.inputs)}), max is 4")
        if len(self.outputs) > 2:
            raise ValueError(f"Too many step outputs ({len(self.outputs)}), max is 2")
        for tid in list(self.inputs) + list(self.outputs):
            if not (0 <= tid <= U16_MAX):
                raise ValueError(f"tensor id out of uint16 range: {tid}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": self.step_type.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }


@dataclass(frozen=True)
class NpuStepDesc(StepDesc):
    """
    NPU step descriptor.

    The step references a contiguous sub-range of the global uops/params blobs,
    expressed in 32-bit words (little-endian).
    """

    uop_off_words: int = 0
    uop_words: int = 0
    param_off_words: int = 0
    param_words: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_type", StepType.NPU)  # type: ignore[misc]
        super().__post_init__()
        for k, v in (
            ("uop_off_words", self.uop_off_words),
            ("uop_words", self.uop_words),
            ("param_off_words", self.param_off_words),
            ("param_words", self.param_words),
        ):
            if v < 0:
                raise ValueError(f"{k} must be non-negative, got {v}")

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update(
            {
                "uop_off_words": self.uop_off_words,
                "uop_words": self.uop_words,
                "param_off_words": self.param_off_words,
                "param_words": self.param_words,
            }
        )
        return d


@dataclass(frozen=True)
class CpuStepDesc(StepDesc):
    """CPU fallback step descriptor (built-in kernels)."""

    kernel: CpuKernel = CpuKernel.ADD
    activation: CpuActivation = CpuActivation.NONE
    # For CONCAT_C, `axis` is implicitly C; keep `axis` for future extension.
    axis: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_type", StepType.CPU)  # type: ignore[misc]
        super().__post_init__()
        if self.kernel == CpuKernel.CONCAT_C and self.axis != 1:
            raise ValueError("CONCAT_C kernel only supports axis==1 (C dimension)")

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({"kernel": self.kernel.name, "activation": self.activation.name, "axis": self.axis})
        return d


@dataclass(frozen=True)
class AliasStepDesc(StepDesc):
    """Zero-copy alias step (identity/reshape)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_type", StepType.ALIAS)  # type: ignore[misc]
        super().__post_init__()
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise ValueError("ALIAS step must have exactly 1 input and 1 output tensor")


@dataclass
class Plan:
    """Full execution plan."""

    tensors: List[TensorDesc] = field(default_factory=list)
    steps: List[StepDesc] = field(default_factory=list)
    arena_bytes: int = 0
    uops_len_words: int = 0
    params_len_words: int = 0
    # Optional quant scale table (v1: per-tensor symmetric scales).
    quant_scales: List[float] = field(default_factory=list)
    tensor_id_by_name: Dict[str, int] = field(default_factory=dict)
    quant_index_by_scale: Dict[float, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "arena_bytes": self.arena_bytes,
            "uops_len_words": self.uops_len_words,
            "params_len_words": self.params_len_words,
            "quant_scales": list(self.quant_scales),
            "tensors": [t.to_dict() for t in self.tensors],
            "steps": [s.to_dict() for s in self.steps],
        }

    def add_tensor(self, desc: TensorDesc) -> None:
        if desc.name in self.tensor_id_by_name:
            raise ValueError(f"Duplicate tensor name in plan: {desc.name}")
        self.tensor_id_by_name[desc.name] = desc.tensor_id
        self.tensors.append(desc)

    def get_tensor_id(self, name: str) -> int:
        if name not in self.tensor_id_by_name:
            raise KeyError(name)
        return self.tensor_id_by_name[name]

    def get_or_add_quant_scale(self, scale: float) -> int:
        """
        Return a stable index for a scalar symmetric quant scale.

        v1 assumes per-tensor scales, so scale is a single float per tensor.
        """
        s = float(scale)
        if s <= 0:
            raise ValueError(f"Invalid quant scale: {scale!r}")
        if s in self.quant_index_by_scale:
            return int(self.quant_index_by_scale[s])
        idx = len(self.quant_scales)
        if idx > U16_MAX:
            raise ValueError("Too many quant scales for v1 (quant_index is uint16).")
        self.quant_scales.append(s)
        self.quant_index_by_scale[s] = idx
        return idx
