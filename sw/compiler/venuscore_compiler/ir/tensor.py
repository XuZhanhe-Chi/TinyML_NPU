# -*- coding: utf-8 -*-
"""
Logical tensor structure for VenusCore IR (VcTensor).

Key fix:
  - get_scale_for_channel() now respects self.q_axis for per-channel quant.
    This prevents mismatch when weights use axis=0 (per-output-channel).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class VcTensor:
    """
    Represents a logical tensor with optional symmetric quant metadata.

    shape is stored as 4D tuple for IR convenience.
    q_axis meaning:
      - For activations (NCHW), per-channel usually uses axis=1.
      - For conv weights (O,I,Kh,Kw), per-output-channel uses axis=0.
    """

    name: str
    shape: tuple[int, int, int, int]
    layout: str = "NCHW"
    dtype: str = "int8"

    # Quant metadata
    scale: float | Sequence[float] | None = None
    q_scheme: Literal["none", "symmetric_per_tensor", "symmetric_per_channel"] = "none"
    q_axis: int | None = None

    # Optional payload/data for debugging or synthetic graphs
    data: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_quantized(self) -> bool:
        return self.q_scheme != "none"

    def is_per_tensor_quant(self) -> bool:
        return self.q_scheme == "symmetric_per_tensor"

    def is_per_channel_quant(self) -> bool:
        return self.q_scheme == "symmetric_per_channel"

    def get_scale_for_channel(self, c: int) -> float:
        """
        Return scale for an index along the quantized axis.

        - per-tensor: returns scalar scale.
        - per-channel: returns scale[c], where c indexes self.q_axis.

        Backward compatibility:
          If q_axis is None, defaults to axis=1 (activation-channel axis in NCHW).
        """
        if not self.is_quantized():
            raise ValueError(f"Tensor {self.name} is not quantized (q_scheme=none).")
        if self.scale is None:
            raise ValueError(f"Tensor {self.name} is missing scale information.")

        if self.is_per_tensor_quant():
            return float(self.scale)  # type: ignore[arg-type]

        if not isinstance(self.scale, Sequence):
            raise TypeError(f"Tensor {self.name} scale must be a sequence for per-channel quantization.")

        axis = 1 if self.q_axis is None else int(self.q_axis)
        if axis < 0 or axis > 3:
            raise ValueError(f"Tensor {self.name} has invalid q_axis={axis}; expected 0..3.")

        expected = int(self.shape[axis])
        if len(self.scale) != expected:
            raise ValueError(
                f"Tensor {self.name} per-channel scale length mismatch: "
                f"len(scale)={len(self.scale)} but shape[axis={axis}]={expected}. "
                f"shape={self.shape}, q_axis={self.q_axis}"
            )

        if c < 0 or c >= len(self.scale):
            raise IndexError(
                f"Index {c} out of range for tensor {self.name} per-channel scales (len={len(self.scale)})."
            )
        return float(self.scale[c])
