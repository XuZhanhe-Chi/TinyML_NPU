# -*- coding: utf-8 -*-
"""
IR normalization and sanity checks for the midend.

Responsibilities:
  - Canonicalize tensor layout tags (upper-case strings: e.g., "NCHW", "NHWC").
  - Require 4D tensors with N==1 for feature maps (batch=1).
  - Validate basic Conv/Depthwise/Pointwise/AvgPool/MaxPool/FC shapes against
    kernel/stride/padding for the supported subset.
"""

from __future__ import annotations

from typing import Any, Tuple

from venuscore_compiler.ir.ops import (
    VcAvgPool,
    VcConv2D,
    VcDepthwiseConv,
    VcPointwiseConv,
    VcFullyConnected,
    VcMaxPool,
    VcOp,
)
from venuscore_compiler.ir.program import VcProgram

__all__ = ["normalize_program"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_nchw(
    shape: Tuple[int, int, int, int],
    layout: str,
) -> Tuple[int, int, int, int]:
    """
    Convert a (4D) shape from the given layout to (N,C,H,W).
    """
    if len(shape) != 4:
        raise ValueError(f"Expected 4D shape, got {shape}.")
    n, d1, d2, d3 = shape
    layout = layout.upper()
    if layout == "NCHW":
        return n, d1, d2, d3
    if layout == "NHWC":
        return n, d3, d1, d2
    raise ValueError(f"Unsupported layout '{layout}'.")


def _normalize_tensor_layouts(program: VcProgram) -> None:
    """
    Canonicalize tensor.layout to upper-case strings and enforce 4D/N==1
    for feature-map tensors (weights/bias may have different leading dim).
    """
    # Rough heuristic: any tensor referenced by an op's weight/bias fields is treated as a param tensor.
    param_names = set()
    for op in program.ops:
        w = getattr(op, "weight", None)
        b = getattr(op, "bias", None)
        if w:
            param_names.add(w)
        if b:
            param_names.add(b)

    for tensor in program.tensors.values():
        layout = getattr(tensor, "layout", "NCHW")
        layout = (layout or "NCHW").upper()
        if layout not in ("NCHW", "NHWC"):
            raise ValueError(
                f"Tensor '{tensor.name}' has unsupported layout '{layout}', "
                "expected 'NCHW' or 'NHWC'."
            )
        tensor.layout = layout

        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) != 4:
            raise ValueError(
                f"Tensor '{tensor.name}' must be 4D (N,C,H,W or N,H,W,C), got {shape}."
            )

        n, _, _, _ = _as_nchw(shape, layout)

        # Param tensors may have N!=1; feature maps require N==1.
        metadata = getattr(tensor, "metadata", {}) or {}
        is_param = (
            tensor.name in param_names
            or ("weight" in tensor.name)
            or ("bias" in tensor.name)
            or bool(metadata.get("param_alias"))
        )
        if not is_param and n != 1:
            raise ValueError(
                f"Tensor '{tensor.name}' batch dimension N must be 1 for feature maps, got N={n}."
            )


def _get_io_tensors(op: VcOp, program: VcProgram):
    if not getattr(op, "inputs", None) or not getattr(op, "outputs", None):
        raise ValueError(f"Op '{op.name}' missing inputs/outputs.")
    try:
        inp = program.tensors[op.inputs[0]]
        out = program.tensors[op.outputs[0]]
    except KeyError as e:  # pragma: no cover - defensive
        raise ValueError(
            f"Op '{op.name}' refers to unknown tensor '{e.args[0]}'."
        ) from e
    return inp, out


# ---------------------------------------------------------------------------
# Op-level shape checks
# ---------------------------------------------------------------------------


def _check_conv_shapes(op: VcConv2D, program: VcProgram) -> None:
    inp, out = _get_io_tensors(op, program)
    n_in, cin, h_in, w_in = _as_nchw(inp.shape, inp.layout)
    n_out, cout, h_out, w_out = _as_nchw(out.shape, out.layout)

    if n_in != 1 or n_out != 1:
        raise ValueError(f"Conv2D '{op.name}' requires N==1 (got N_in={n_in}, N_out={n_out}).")
    if cin <= 0 or cout <= 0:
        raise ValueError(f"Conv2D '{op.name}' invalid Cin/Cout ({cin},{cout}).")

    kh = getattr(op, "kernel_h", None)
    kw = getattr(op, "kernel_w", None)
    sh = getattr(op, "stride_h", None)
    sw = getattr(op, "stride_w", None)
    pt = getattr(op, "pad_top", 0)
    pb = getattr(op, "pad_bottom", 0)
    pl = getattr(op, "pad_left", 0)
    pr = getattr(op, "pad_right", 0)

    if kh is None or kw is None or sh is None or sw is None:
        raise ValueError(f"Conv2D '{op.name}' missing kernel/stride information.")

    # Check output H/W against standard conv formula (for sanity)
    h_eff = h_in + pt + pb
    w_eff = w_in + pl + pr
    h_exp = (h_eff - kh) // sh + 1
    w_exp = (w_eff - kw) // sw + 1
    if h_exp != h_out or w_exp != w_out:
        raise ValueError(
            f"Conv2D '{op.name}' output shape mismatch: expected ({h_exp},{w_exp}), "
            f"got ({h_out},{w_out})."
        )


def _check_depthwise_shapes(op: VcDepthwiseConv, program: VcProgram) -> None:
    inp, out = _get_io_tensors(op, program)
    n_in, cin, h_in, w_in = _as_nchw(inp.shape, inp.layout)
    n_out, cout, h_out, w_out = _as_nchw(out.shape, out.layout)

    if n_in != 1 or n_out != 1:
        raise ValueError(f"DepthwiseConv '{op.name}' requires N==1.")
    if cin <= 0 or cout <= 0:
        raise ValueError(f"DepthwiseConv '{op.name}' invalid Cin/Cout ({cin},{cout}).")

    kh = getattr(op, "kernel_h", None)
    kw = getattr(op, "kernel_w", None)
    sh = getattr(op, "stride_h", None)
    sw = getattr(op, "stride_w", None)
    pt = getattr(op, "pad_top", 0)
    pb = getattr(op, "pad_bottom", 0)
    pl = getattr(op, "pad_left", 0)
    pr = getattr(op, "pad_right", 0)

    if kh is None or kw is None or sh is None or sw is None:
        raise ValueError(f"DepthwiseConv '{op.name}' missing kernel/stride information.")

    h_eff = h_in + pt + pb
    w_eff = w_in + pl + pr
    h_exp = (h_eff - kh) // sh + 1
    w_exp = (w_eff - kw) // sw + 1
    if h_exp != h_out or w_exp != w_out:
        raise ValueError(
            f"DepthwiseConv '{op.name}' output shape mismatch: expected ({h_exp},{w_exp}), "
            f"got ({h_out},{w_out})."
        )


def _check_pointwise_shapes(op: VcPointwiseConv, program: VcProgram) -> None:
    inp, out = _get_io_tensors(op, program)
    n_in, cin, h_in, w_in = _as_nchw(inp.shape, inp.layout)
    n_out, cout, h_out, w_out = _as_nchw(out.shape, out.layout)

    if n_in != 1 or n_out != 1:
        raise ValueError(f"PointwiseConv '{op.name}' requires N==1.")
    if cin <= 0 or cout <= 0:
        raise ValueError(f"PointwiseConv '{op.name}' invalid Cin/Cout ({cin},{cout}).")

    kh = getattr(op, "kernel_h", None)
    kw = getattr(op, "kernel_w", None)
    sh = getattr(op, "stride_h", None)
    sw = getattr(op, "stride_w", None)
    pt = getattr(op, "pad_top", 0)
    pb = getattr(op, "pad_bottom", 0)
    pl = getattr(op, "pad_left", 0)
    pr = getattr(op, "pad_right", 0)

    if kh is None or kw is None or sh is None or sw is None:
        raise ValueError(f"PointwiseConv '{op.name}' missing kernel/stride information.")

    h_eff = h_in + pt + pb
    w_eff = w_in + pl + pr
    h_exp = (h_eff - kh) // sh + 1
    w_exp = (w_eff - kw) // sw + 1
    if h_exp != h_out or w_exp != w_out:
        raise ValueError(
            f"PointwiseConv '{op.name}' output shape mismatch: expected ({h_exp},{w_exp}), "
            f"got ({h_out},{w_out})."
        )


def _check_avgpool_shapes(op: VcAvgPool, program: VcProgram) -> None:
    inp, out = _get_io_tensors(op, program)
    n_in, cin, h_in, w_in = _as_nchw(inp.shape, inp.layout)
    n_out, cout, h_out, w_out = _as_nchw(out.shape, out.layout)

    if n_in != 1 or n_out != 1:
        raise ValueError(f"AvgPool '{op.name}' requires N==1.")
    if cin != cout:
        raise ValueError(f"AvgPool '{op.name}' expects Cin==Cout, got {cin},{cout}.")

    kh = getattr(op, "kernel_h", None)
    kw = getattr(op, "kernel_w", None)
    sh = getattr(op, "stride_h", None)
    sw = getattr(op, "stride_w", None)
    pt = getattr(op, "pad_top", 0)
    pb = getattr(op, "pad_bottom", 0)
    pl = getattr(op, "pad_left", 0)
    pr = getattr(op, "pad_right", 0)

    if kh is None or kw is None or sh is None or sw is None:
        raise ValueError(f"AvgPool '{op.name}' missing kernel/stride information.")
    if kh != 2 or kw != 2:
        raise ValueError(f"AvgPool '{op.name}' kernel must be 2x2, got {kh}x{kw}.")

    h_eff = h_in + pt + pb
    w_eff = w_in + pl + pr
    h_exp = (h_eff - kh) // sh + 1
    w_exp = (w_eff - kw) // sw + 1
    if h_exp != h_out or w_exp != w_out:
        raise ValueError(
            f"AvgPool '{op.name}' output shape mismatch: expected ({h_exp},{w_exp}), "
            f"got ({h_out},{w_out})."
        )


def _check_maxpool_shapes(op: VcMaxPool, program: VcProgram) -> None:
    inp, out = _get_io_tensors(op, program)
    n_in, cin, h_in, w_in = _as_nchw(inp.shape, inp.layout)
    n_out, cout, h_out, w_out = _as_nchw(out.shape, out.layout)

    if n_in != 1 or n_out != 1:
        raise ValueError(f"MaxPool '{op.name}' requires N==1.")
    if cin != cout:
        raise ValueError(f"MaxPool '{op.name}' expects Cin==Cout, got {cin},{cout}.")

    kh = getattr(op, "kernel_h", None)
    kw = getattr(op, "kernel_w", None)
    sh = getattr(op, "stride_h", None)
    sw = getattr(op, "stride_w", None)
    pt = getattr(op, "pad_top", 0)
    pb = getattr(op, "pad_bottom", 0)
    pl = getattr(op, "pad_left", 0)
    pr = getattr(op, "pad_right", 0)

    if kh is None or kw is None or sh is None or sw is None:
        raise ValueError(f"MaxPool '{op.name}' missing kernel/stride information.")
    if kh != 2 or kw != 2:
        raise ValueError(f"MaxPool '{op.name}' kernel must be 2x2, got {kh}x{kw}.")

    h_eff = h_in + pt + pb
    w_eff = w_in + pl + pr
    h_exp = (h_eff - kh) // sh + 1
    w_exp = (w_eff - kw) // sw + 1
    if h_exp != h_out or w_exp != w_out:
        raise ValueError(
            f"MaxPool '{op.name}' output shape mismatch: expected ({h_exp},{w_exp}), "
            f"got ({h_out},{w_out})."
        )


def _check_fc_shapes(op: VcFullyConnected, program: VcProgram) -> None:
    inp, out = _get_io_tensors(op, program)
    n_in, cin, h_in, w_in = _as_nchw(inp.shape, inp.layout)
    n_out, cout, h_out, w_out = _as_nchw(out.shape, out.layout)

    if n_in != 1 or n_out != 1:
        raise ValueError(f"FullyConnected '{op.name}' requires N==1.")
    if h_in != 1 or w_in != 1 or h_out != 1 or w_out != 1:
        raise ValueError(
            f"FullyConnected '{op.name}' expects H=W=1 for input/output "
            f"(got H_in={h_in}, W_in={w_in}, H_out={h_out}, W_out={w_out})."
        )
    if cin <= 0 or cout <= 0:
        raise ValueError(f"FullyConnected '{op.name}' invalid Cin/Cout ({cin},{cout}).")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_program(program: VcProgram) -> VcProgram:
    """
    Normalize the IR in-place and perform basic shape checks.

    - Canonicalize tensor layouts (NCHW/NHWC).
    - Enforce batch N==1 for feature maps.
    - Validate Conv/Depthwise/Pointwise/AvgPool/MaxPool/FC shapes against standard
      mathematical formulas (no hardware-specific constraints here).
    """
    _normalize_tensor_layouts(program)

    for op in program.ops:
        if isinstance(op, VcConv2D):
            _check_conv_shapes(op, program)
        elif isinstance(op, VcDepthwiseConv):
            _check_depthwise_shapes(op, program)
        elif isinstance(op, VcPointwiseConv):
            _check_pointwise_shapes(op, program)
        elif isinstance(op, VcAvgPool):
            _check_avgpool_shapes(op, program)
        elif isinstance(op, VcMaxPool):
            _check_maxpool_shapes(op, program)
        elif isinstance(op, VcFullyConnected):
            _check_fc_shapes(op, program)
        else:
            # Unsupported ops are ignored by this normalization pass.
            continue

    # Lower FullyConnected to PointwiseConv so the conv-like backend pipeline can compile it.
    _lower_fully_connected_to_pointwise(program)
    return program


def _lower_fully_connected_to_pointwise(program: VcProgram) -> None:
    """
    Rewrite VcFullyConnected ops into VcPointwiseConv ops.

    This is a semantic equivalence for N==1 with flattened feature maps:
      - IFM:  [1, Cin, 1, 1]
      - OFM:  [1, Cout, 1, 1]
      - W:    [Cout, Cin, 1, 1]
    """
    if not program.ops:
        return

    new_ops: list[VcOp] = []
    for op in program.ops:
        if not isinstance(op, VcFullyConnected):
            new_ops.append(op)
            continue

        if not op.inputs or not op.outputs:
            raise ValueError(f"FullyConnected '{op.name}' missing inputs/outputs.")
        if not op.weight:
            raise ValueError(f"FullyConnected '{op.name}' missing weight tensor name.")

        ifm = program.tensors.get(op.inputs[0])
        ofm = program.tensors.get(op.outputs[0])
        if ifm is None or ofm is None:
            raise ValueError(f"FullyConnected '{op.name}' refers to missing IFM/OFM tensor.")

        _, cin, h_in, w_in = _as_nchw(ifm.shape, ifm.layout)
        _, cout, h_out, w_out = _as_nchw(ofm.shape, ofm.layout)
        if h_in != 1 or w_in != 1 or h_out != 1 or w_out != 1:
            raise ValueError(
                f"FullyConnected '{op.name}' lowering expects H=W=1 for IFM/OFM, "
                f"got IFM(H,W)=({h_in},{w_in}) OFM(H,W)=({h_out},{w_out})."
            )

        weight_tensor = program.tensors.get(op.weight)
        if weight_tensor is None:
            raise ValueError(f"FullyConnected '{op.name}' refers to missing weight tensor '{op.weight}'.")

        _reshape_fc_weight_to_conv1x1_inplace(weight_tensor=weight_tensor, cin=cin, cout=cout)

        pw = VcPointwiseConv(
            name=op.name,
            inputs=list(op.inputs),
            outputs=list(op.outputs),
            weight=op.weight,
            bias=op.bias,
            activation=op.activation,
            qmode=op.qmode,
        )
        new_ops.append(pw)

    program.ops = new_ops


def _reshape_fc_weight_to_conv1x1_inplace(weight_tensor: Any, cin: int, cout: int) -> None:
    """
    Normalize FC weight into [Cout, Cin, 1, 1] with flattened row-major data.

    Frontends commonly represent FC weight as a 2D matrix [Cout, Cin]. Some
    models may provide the transposed shape [Cin, Cout]. We normalize both.
    """
    shape = getattr(weight_tensor, "shape", None)
    data = getattr(weight_tensor, "data", None)
    name = getattr(weight_tensor, "name", "<unknown>")

    if shape is None or len(shape) != 4:
        raise ValueError(f"FC weight '{name}' must be 4D, got {shape}.")

    s0, s1, s2, s3 = (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
    if s2 != 1 or s3 != 1:
        raise ValueError(f"FC weight '{name}' must have Kh=Kw=1, got shape={shape}.")

    transpose = False
    if s0 == cout and s1 == cin:
        transpose = False
    elif s0 == cin and s1 == cout:
        transpose = True
    else:
        raise ValueError(
            f"FC weight '{name}' shape must match (Cout,Cin) or (Cin,Cout): cin={cin}, cout={cout}, shape={shape}."
        )

    # Normalize shape to [Cout, Cin, 1, 1].
    weight_tensor.shape = (int(cout), int(cin), 1, 1)

    if data is None:
        return

    flat: list[int] = []
    if transpose:
        for co in range(cout):
            for ci in range(cin):
                flat.append(int(_get_weight_2d(data, ci, co)))
    else:
        for co in range(cout):
            for ci in range(cin):
                flat.append(int(_get_weight_2d(data, co, ci)))

    if len(flat) != cout * cin:
        raise ValueError(f"FC weight '{name}' flattened size mismatch: len={len(flat)} expected={cout * cin}.")

    # Use a flattened buffer so backend packing can rely on linear indexing.
    weight_tensor.data = flat


def _get_weight_2d(data: Any, i: int, j: int) -> int:
    """Index a 2D weight matrix stored as nested lists."""
    try:
        v = data[i][j]
        # Common cases:
        #  - data is 2D: data[co][ci] -> int
        #  - data is 4D with trailing [1][1]: data[co][ci][0][0] -> int
        if isinstance(v, (int, float)):
            return int(v)
        return int(v[0][0])
    except Exception as exc:
        raise ValueError(f"Expected 2D nested weight data, failed at ({i},{j}).") from exc
