# -*- coding: utf-8 -*-
"""
Module overview:
  - Defines VenusCore IR operation nodes such as convolutions, pooling, and fully connected layers.
  - Dependencies:
    * Depends on: dataclasses.
    * Used by: ir.program, frontend builders, and backend lowering/codegen passes.
  - Hardware-free layer: these ops describe logical intent; backend modules handle tiling, addresses,
    and VenusCore uOP encoding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class VcOp:
    """
    Base class for all VenusCore IR operations.

    Field semantics:
      - inputs / outputs: logical data tensor names in the IR program (referencing VcTensor objects).
        Parameter tensors should not be listed here.
      - weight / bias: optional parameter tensor names (e.g., convolution or fully connected weights/bias).
      - activation: activation hint (e.g., "relu", "relu6", "none"); allowed values are refined by backend rules.
      - qmode: quantization mode tag (e.g., "symmetric_int8_per_tensor", "symmetric_int8_per_channel");
        the precise set is backend-defined and may evolve; may be None in early IR before quantization is decided.
      - op_type: short tag used for debugging/dispatch; subclasses override with specific identifiers
        such as "conv2d", "depthwise_conv2d", "avg_pool2d", "fully_connected".

    Quantization parameters (scales) live on tensors; requantization multipliers/shifts live in backend
    parameter blocks. No hardware/uOP-specific fields should be added here.
    """

    name: str
    inputs: list[str]
    outputs: list[str]
    weight: Optional[str] = None
    bias: Optional[str] = None
    activation: Optional[str] = None
    qmode: Optional[str] = None
    op_type: str = field(init=False, default="generic")

    def to_dict(self) -> dict:
        """Convert the op to a plain dictionary for debugging/serialization (e.g., debug dumps or JSON IR)."""

        return asdict(self)


@dataclass
class VcConv2D(VcOp):
    """
    Standard 2D convolution.

    Tuple parameters (kernel, stride, padding, dilation) are the source of truth; scalar fields are
    derived in __post_init__ for backend convenience.
    """

    kernel: tuple[int, int] = (3, 3)
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)  # top, bottom, left, right
    dilation: tuple[int, int] = (1, 1)
    kernel_h: int = 3
    kernel_w: int = 3
    stride_h: int = 1
    stride_w: int = 1
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    groups: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    op_type: str = field(init=False, default="conv2d")

    def __post_init__(self) -> None:
        # Tuple fields take precedence over scalars when both are set; scalars are derived for backend convenience.
        if self.kernel:
            self.kernel_h, self.kernel_w = self.kernel
        if self.stride:
            self.stride_h, self.stride_w = self.stride
        if self.padding:
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = self.padding
        if self.dilation:
            self.dilation_h, self.dilation_w = self.dilation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VcDepthwiseConv(VcOp):
    """
    Depthwise convolution (one filter per input channel).

    Tuple parameters are the source of truth; scalar fields are derived in __post_init__.
    groups == 0 means it will be inferred later (typically groups == input channels) during legalization
    before backend codegen.
    """

    kernel: tuple[int, int] = (3, 3)
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    dilation: tuple[int, int] = (1, 1)
    kernel_h: int = 3
    kernel_w: int = 3
    stride_h: int = 1
    stride_w: int = 1
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    groups: int = 0  # 0 => infer groups == input channels during a legalization pass.
    op_type: str = field(init=False, default="depthwise_conv")

    def __post_init__(self) -> None:
        # Tuple fields take precedence; scalar fields are derived for backend convenience.
        if self.kernel:
            self.kernel_h, self.kernel_w = self.kernel
        if self.stride:
            self.stride_h, self.stride_w = self.stride
        if self.padding:
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = self.padding
        if self.dilation:
            self.dilation_h, self.dilation_w = self.dilation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VcPointwiseConv(VcOp):
    """
    1x1 convolution often used for channel mixing.

    Tuple parameters are the source of truth; scalar fields are derived in __post_init__.
    """

    kernel: tuple[int, int] = (1, 1)
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    dilation: tuple[int, int] = (1, 1)
    kernel_h: int = 1
    kernel_w: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    groups: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    op_type: str = field(init=False, default="pointwise_conv")

    def __post_init__(self) -> None:
        # Tuple fields take precedence; scalar fields are derived for backend convenience.
        if self.kernel:
            self.kernel_h, self.kernel_w = self.kernel
        if self.stride:
            self.stride_h, self.stride_w = self.stride
        if self.padding:
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = self.padding
        if self.dilation:
            self.dilation_h, self.dilation_w = self.dilation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VcAvgPool(VcOp):
    """
    Average pooling operation.

    Tuple parameters are the source of truth; scalar fields are derived in __post_init__.
    """

    kernel: tuple[int, int] = (2, 2)
    stride: tuple[int, int] = (2, 2)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    kernel_h: int = 2
    kernel_w: int = 2
    stride_h: int = 2
    stride_w: int = 2
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    op_type: str = field(init=False, default="avg_pool")

    def __post_init__(self) -> None:
        # Tuple fields take precedence; scalar fields are derived for backend convenience.
        if self.kernel:
            self.kernel_h, self.kernel_w = self.kernel
        if self.stride:
            self.stride_h, self.stride_w = self.stride
        if self.padding:
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = self.padding

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VcMaxPool(VcOp):
    """
    Max pooling operation.

    Tuple parameters are the source of truth; scalar fields are derived in __post_init__.
    """

    kernel: tuple[int, int] = (2, 2)
    stride: tuple[int, int] = (2, 2)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    kernel_h: int = 2
    kernel_w: int = 2
    stride_h: int = 2
    stride_w: int = 2
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    op_type: str = field(init=False, default="max_pool")

    def __post_init__(self) -> None:
        if self.kernel:
            self.kernel_h, self.kernel_w = self.kernel
        if self.stride:
            self.stride_h, self.stride_w = self.stride
        if self.padding:
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = self.padding

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VcFullyConnected(VcOp):
    """Fully-connected layer."""

    op_type: str = field(init=False, default="fully_connected")

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# CPU fallback + view (alias) ops used by the Plan pipeline
# -----------------------------------------------------------------------------


@dataclass
class VcAdd(VcOp):
    """Elementwise add (CPU fallback in plan v1)."""

    op_type: str = field(init=False, default="add")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VcConcatC(VcOp):
    """
    Concat along channel dimension (C in NCHW).

    The plan v1 only supports concat on axis==1 (C).
    """

    axis: int = 1
    op_type: str = field(init=False, default="concat")

    def __post_init__(self) -> None:
        if self.axis != 1:
            raise ValueError(f"VcConcatC only supports axis==1 (C), got axis={self.axis}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VcIdentity(VcOp):
    """Identity/view op (zero-copy alias in plan v1)."""

    op_type: str = field(init=False, default="identity")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VcReshape(VcOp):
    """Reshape/view op (zero-copy alias in plan v1)."""

    new_shape: Optional[tuple[int, ...]] = None
    op_type: str = field(init=False, default="reshape")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VcFlatten(VcOp):
    """Flatten/view op (zero-copy alias in plan v1)."""

    axis: int = 1
    op_type: str = field(init=False, default="flatten")

    def to_dict(self) -> dict:
        return asdict(self)
