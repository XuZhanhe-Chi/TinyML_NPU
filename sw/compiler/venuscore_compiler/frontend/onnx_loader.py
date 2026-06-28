# -*- coding: utf-8 -*-
"""
ONNX loader for VenusCore IR (VcProgram).

Supported ops (strict):
  - Conv: 1x1(PW), 3x3(Conv), 3x3(DW multiplier=1)
  - AveragePool / GlobalAveragePool
  - MaxPool (2x2, stride=2, pad 0/1)
  - Gemm -> FullyConnected
  - Relu: must be fuseable into previous compute op
  - View nodes (no compute op emitted; tracked by storage_alias):
      QuantizeLinear / DequantizeLinear / Identity / Flatten / Reshape

Strictness:
  - Any unsupported op or unsupported attribute combination -> print error + raise ValueError.
  - No silent skipping.

Notes:
  - Assumes NCHW-like semantics for 4D tensors.
  - Quantization: expects symmetric (zero_point == 0 when provided).
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Any, Callable

try:
    import onnx  # type: ignore
    import numpy as np  # type: ignore
    from onnx import AttributeProto  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "onnx_loader requires the 'onnx' and 'numpy' packages. "
        "Install with `pip install onnx numpy`."
    ) from exc

from venuscore_compiler.ir.ops import (
    VcAdd,
    VcAvgPool,
    VcMaxPool,
    VcConcatC,
    VcConv2D,
    VcDepthwiseConv,
    VcFlatten,
    VcIdentity,
    VcPointwiseConv,
    VcFullyConnected,
    VcReshape,
)
from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.ir.tensor import VcTensor

__all__ = ["load_onnx_to_ir"]


def load_onnx_to_ir(path: str) -> VcProgram:
    model = onnx.load(path)
    graph = model.graph

    init_map: Dict[str, np.ndarray] = {init.name: _to_numpy(init) for init in graph.initializer}
    const_map: Dict[str, np.ndarray] = _collect_constant_nodes(graph)

    # Pre-scan Q/DQ metadata.
    qdq = _collect_qdq_info(graph, init_map, const_map)
    q_scales: Dict[str, np.ndarray] = qdq["q_scales"]
    q_axis: Dict[str, int] = qdq["q_axis"]
    dq_scales: Dict[str, np.ndarray] = qdq["dq_scales"]
    dq_axis: Dict[str, int] = qdq["dq_axis"]

    program = VcProgram(path)

    tensors: Dict[str, VcTensor] = {}
    tensor_shapes: Dict[str, Tuple[int, ...]] = {}

    # tensor_name -> tensor_name (same underlying storage/buffer)
    storage_alias: Dict[str, str] = {}

    # Seed shapes from graph inputs/outputs/value_info
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        shape = _get_shape_from_value(value)
        if shape is None:
            continue
        tensor_shapes[value.name] = tuple(max(1, int(x)) for x in shape)

    graph_inputs = {vi.name for vi in graph.input}
    init_names = set(init_map.keys())

    def resolve_storage(name: str) -> str:
        """Follow storage_alias chain to its root. Detect cycles."""
        seen = set()
        cur = name
        while cur in storage_alias:
            if cur in seen:
                _fail(None, f"storage_alias cycle detected at '{cur}' (start='{name}'), chain={sorted(seen)}")
            seen.add(cur)
            cur = storage_alias[cur]
        return cur

    def ensure_tensor(
        name: str,
        shape: Tuple[int, ...] | None = None,
        dtype: str = "int8",
        layout: str = "NCHW",
    ) -> VcTensor:
        """Get or create a VcTensor with best-effort shape + quant metadata."""
        if name in tensors:
            t = tensors[name]
            if shape is not None:
                new_shape = _force_4d(_sanitize_shape(shape))
                if tuple(t.shape) != new_shape:
                    t.shape = new_shape
                    tensor_shapes[name] = new_shape
            return t

        inferred_shape = shape or tensor_shapes.get(name) or _find_shape(graph, name)
        if inferred_shape is None:
            if name in init_map:
                inferred_shape = init_map[name].shape
            elif name in const_map:
                inferred_shape = const_map[name].shape
            else:
                inferred_shape = (1, 1, 1, 1)

        inferred_shape = _force_4d(_sanitize_shape(inferred_shape))

        # Quant metadata
        q_scheme = "none"
        q_axis_val: Optional[int] = None
        scale: Any = None

        # Prefer DQ then Q
        if name in dq_scales:
            scale, q_scheme, q_axis_val = _scale_to_ir(dq_scales[name], dq_axis.get(name, 0))
        elif name in q_scales:
            scale, q_scheme, q_axis_val = _scale_to_ir(q_scales[name], q_axis.get(name, 0))

        tensor = VcTensor(
            name=name,
            shape=tuple(int(x) for x in inferred_shape),
            layout=layout,
            dtype=dtype,
            scale=scale,
            q_scheme=q_scheme,
            q_axis=q_axis_val,
        )
        tensors[name] = tensor
        tensor_shapes[name] = tuple(int(x) for x in inferred_shape)
        program.add_tensor(tensor)
        return tensor

    # Create tensors for initializers
    for init_name, arr in init_map.items():
        t = ensure_tensor(init_name, shape=arr.shape, dtype=_np_dtype_to_ir(arr.dtype))
        try:
            t.data = arr.tolist()
        except Exception:
            pass

    mapped_nodes: List[str] = []

    for node in graph.node:
        op_type = node.op_type
        node_id = node.name or (node.output[0] if node.output else "<unnamed>")

        if op_type == "Constant":
            mapped_nodes.append(f"{node_id}:Constant")
            continue

        if op_type in ("QuantizeLinear", "DequantizeLinear", "Identity", "Flatten", "Reshape"):
            _handle_view_node(
                graph=graph,
                node=node,
                graph_inputs=graph_inputs,
                init_map=init_map,
                const_map=const_map,
                ensure_tensor=ensure_tensor,
                tensor_shapes=tensor_shapes,
                storage_alias=storage_alias,
                program=program,
                fail=_fail,
            )
            mapped_nodes.append(f"{node_id}:{op_type}")
            continue

        if op_type == "Relu":
            if len(node.input) != 1 or len(node.output) != 1:
                _fail(node, f"Relu expects 1 input/1 output, got inputs={list(node.input)}, outputs={list(node.output)}")
            relu_in = node.input[0]
            relu_out = node.output[0]
            fused = _maybe_fuse_relu(program, relu_in, relu_out, resolve_storage)
            if not fused:
                _fail(node, f"Relu not fuseable; last_op must output '{relu_in}'. Standalone Relu not supported.")
            in_shape = tensor_shapes.get(relu_in) or _find_shape(graph, relu_in)
            if in_shape is not None:
                tensor_shapes[relu_out] = tuple(int(x) for x in in_shape)
                ensure_tensor(relu_out, shape=in_shape)
            mapped_nodes.append(f"{node_id}:Relu(fused)")
            continue

        if op_type == "Clip":
            # Some QDQ graphs use Clip(0,6) to express ReLU6.
            # We only support Clip(0,6) and require it to be fuseable into the
            # preceding compute op to avoid producing disconnected IR graphs.
            if len(node.input) < 1 or len(node.output) != 1:
                _fail(
                    node,
                    f"Clip expects >=1 input and 1 output, got inputs={list(node.input)}, outputs={list(node.output)}",
                )
            clip_in = node.input[0]
            clip_out = node.output[0]
            clip_min, clip_max = _get_clip_min_max(node, init_map, const_map, _fail)
            if not (clip_min == 0.0 and clip_max == 6.0):
                _fail(
                    node,
                    f"Clip bounds not supported: min={clip_min}, max={clip_max} (only Clip(0,6) is supported as ReLU6).",
                )
            fused = _maybe_fuse_activation(program, clip_in, clip_out, resolve_storage, act="relu6")
            if not fused:
                _fail(
                    node,
                    f"Clip(0,6) not fuseable; last_op must output '{clip_in}'. Standalone Clip not supported.",
                )
            in_shape = tensor_shapes.get(clip_in) or _find_shape(graph, clip_in)
            if in_shape is not None:
                tensor_shapes[clip_out] = tuple(int(x) for x in in_shape)
                ensure_tensor(clip_out, shape=in_shape)
            mapped_nodes.append(f"{node_id}:Clip(0,6)->ReLU6(fused)")
            continue

        if op_type == "Conv":
            _handle_conv(
                graph=graph,
                node=node,
                node_id=node_id,
                ensure_tensor=ensure_tensor,
                tensor_shapes=tensor_shapes,
                init_map=init_map,
                const_map=const_map,
                storage_alias=storage_alias,
                program=program,
                resolve_storage=resolve_storage,
                fail=_fail,
            )
            mapped_nodes.append(f"{node_id}:Conv")
            continue

        if op_type in ("AveragePool", "GlobalAveragePool"):
            _handle_avgpool(
                graph=graph,
                node=node,
                node_id=node_id,
                ensure_tensor=ensure_tensor,
                tensor_shapes=tensor_shapes,
                program=program,
                resolve_storage=resolve_storage,
                fail=_fail,
            )
            mapped_nodes.append(f"{node_id}:{op_type}")
            continue

        if op_type in ("MaxPool", "GlobalMaxPool"):
            _handle_maxpool(
                graph=graph,
                node=node,
                node_id=node_id,
                ensure_tensor=ensure_tensor,
                tensor_shapes=tensor_shapes,
                program=program,
                resolve_storage=resolve_storage,
                fail=_fail,
            )
            mapped_nodes.append(f"{node_id}:{op_type}")
            continue

        if op_type == "Gemm":
            _handle_gemm_as_fc(
                graph=graph,
                node=node,
                node_id=node_id,
                ensure_tensor=ensure_tensor,
                tensor_shapes=tensor_shapes,
                init_map=init_map,
                const_map=const_map,
                storage_alias=storage_alias,
                program=program,
                resolve_storage=resolve_storage,
                fail=_fail,
            )
            mapped_nodes.append(f"{node_id}:Gemm->FC")
            continue

        if op_type == "Add":
            if len(node.input) != 2 or len(node.output) != 1:
                _fail(node, f"Add expects 2 inputs and 1 output, got inputs={list(node.input)}, outputs={list(node.output)}")
            a_name, b_name = node.input[0], node.input[1]
            y_name = node.output[0]
            a_root = resolve_storage(a_name)
            b_root = resolve_storage(b_name)
            y_root = resolve_storage(y_name)

            a_shape = tensor_shapes.get(a_root) or tensor_shapes.get(a_name) or _find_shape(graph, a_name)
            b_shape = tensor_shapes.get(b_root) or tensor_shapes.get(b_name) or _find_shape(graph, b_name)
            if a_shape is None or b_shape is None:
                _fail(node, f"Add requires known input shapes, got a={a_shape}, b={b_shape}")
            a4 = _force_4d(_sanitize_shape(a_shape))
            b4 = _force_4d(_sanitize_shape(b_shape))
            if a4 != b4:
                _fail(node, f"Add broadcasting not supported: a={a4}, b={b4}")
            tensor_shapes[y_root] = a4

            ta = ensure_tensor(a_root, shape=a4, dtype="int8")
            tb = ensure_tensor(b_root, shape=b4, dtype="int8")
            if ta.dtype != "int8" or tb.dtype != "int8":
                _fail(node, f"Add only supports int8 tensors (got a={ta.dtype}, b={tb.dtype})")
            # Allow mismatched per-tensor scales; a scale-aware CPU add kernel can requantize.
            # The output scale is often defined later by QuantizeLinear on this value.
            ty = ensure_tensor(y_root, shape=a4, dtype="int8")
            ty.q_scheme = ta.q_scheme
            ty.q_axis = ta.q_axis

            program.add_op(VcAdd(name=node_id, inputs=[a_root, b_root], outputs=[y_root]))
            mapped_nodes.append(f"{node_id}:Add->VcAdd(CPU)")
            continue

        if op_type == "Concat":
            if len(node.input) < 2 or len(node.output) != 1:
                _fail(node, f"Concat expects >=2 inputs and 1 output, got inputs={list(node.input)}, outputs={list(node.output)}")
            axis = int(_get_attr(node, "axis", 1))
            if axis != 1:
                _fail(node, f"Concat only supports axis==1 (C), got axis={axis}")
            inputs = list(node.input)
            y_name = node.output[0]
            roots = [resolve_storage(x) for x in inputs]
            y_root = resolve_storage(y_name)

            shapes = []
            tensors_in = []
            for x_name, x_root in zip(inputs, roots):
                x_shape = tensor_shapes.get(x_root) or tensor_shapes.get(x_name) or _find_shape(graph, x_name)
                if x_shape is None:
                    _fail(node, f"Concat requires known input shape for '{x_name}'")
                x4 = _force_4d(_sanitize_shape(x_shape))
                shapes.append(x4)
                tensors_in.append(ensure_tensor(x_root, shape=x4, dtype="int8"))

            n0, c0, h0, w0 = shapes[0]
            c_sum = 0
            for x4 in shapes:
                n, c, h, w = x4
                if (n, h, w) != (n0, h0, w0):
                    _fail(node, f"Concat_C requires same N/H/W across inputs, got {shapes}")
                c_sum += c
            out_shape = (n0, c_sum, h0, w0)
            tensor_shapes[y_root] = out_shape

            t0 = tensors_in[0]
            for tx in tensors_in[1:]:
                if getattr(tx, "q_scheme", "none") != getattr(t0, "q_scheme", "none") or getattr(tx, "scale", None) != getattr(t0, "scale", None):
                    _fail(node, f"Concat_C requires quant-aligned inputs (q_scheme/scale must match): '{t0.name}' vs '{tx.name}'")

            ty = ensure_tensor(y_root, shape=out_shape, dtype="int8")
            ty.scale = t0.scale
            ty.q_scheme = t0.q_scheme
            ty.q_axis = t0.q_axis

            program.add_op(VcConcatC(name=node_id, inputs=roots, outputs=[y_root], axis=1))
            mapped_nodes.append(f"{node_id}:Concat(axis=1)->VcConcatC(CPU)")
            continue

        _fail(node, f"Unsupported ONNX op_type='{op_type}'")

    program.metadata["onnx_mapped_nodes"] = mapped_nodes
    program.metadata["storage_alias"] = dict(storage_alias)

    _check_connectivity(
        program=program,
        graph_inputs=graph_inputs,
        init_names=init_names,
        resolve_storage=resolve_storage,
    )

    program.validate()
    return program


# -----------------------------------------------------------------------------
# View nodes (Q/DQ/Identity/Flatten/Reshape)
# -----------------------------------------------------------------------------

def _handle_view_node(
    graph,
    node,
    graph_inputs: set[str],
    init_map: Dict[str, np.ndarray],
    const_map: Dict[str, np.ndarray],
    ensure_tensor,
    tensor_shapes: Dict[str, Tuple[int, ...]],
    storage_alias: Dict[str, str],
    program: VcProgram,
    fail,
) -> None:
    op_type = node.op_type

    def get_const(name: str) -> Optional[np.ndarray]:
        if name in init_map:
            return init_map[name]
        if name in const_map:
            return const_map[name]
        return None

    if op_type == "Identity":
        if len(node.input) != 1 or len(node.output) != 1:
            fail(node, f"Identity expects 1 input/1 output, got inputs={list(node.input)}, outputs={list(node.output)}")
        x, y = node.input[0], node.output[0]
        x_shape = tensor_shapes.get(x) or _find_shape(graph, x)
        if x_shape is not None:
            tensor_shapes[y] = tuple(int(v) for v in x_shape)
        t_in = ensure_tensor(x, shape=tensor_shapes.get(x))
        t_out = ensure_tensor(y, shape=tensor_shapes.get(y), dtype=t_in.dtype)
        t_out.scale = t_in.scale
        t_out.q_scheme = t_in.q_scheme
        t_out.q_axis = t_in.q_axis
        storage_alias[y] = x
        program.add_op(VcIdentity(name=node.name or f"Identity_{y}", inputs=[x], outputs=[y]))
        return

    if op_type in ("QuantizeLinear", "DequantizeLinear"):
        if len(node.input) < 2 or len(node.output) != 1:
            fail(node, f"{op_type} expects >=2 inputs and 1 output, got inputs={list(node.input)}, outputs={list(node.output)}")
        x = node.input[0]
        scale_name = node.input[1]
        zp_name = node.input[2] if len(node.input) >= 3 else None
        y = node.output[0]

        scale_arr = get_const(scale_name)
        if scale_arr is None:
            fail(node, f"{op_type} scale must be initializer/constant, got '{scale_name}'")

        if zp_name:
            zp_arr = get_const(zp_name)
            if zp_arr is None:
                fail(node, f"{op_type} zero_point must be initializer/constant, got '{zp_name}'")
            if np.any(zp_arr != 0):
                fail(node, f"{op_type} requires symmetric int8 (zero_point==0). Got '{zp_name}' values={zp_arr.flatten()[:8].tolist()}...")

        axis = int(_get_attr(node, "axis", 0))

        x_shape = tensor_shapes.get(x) or _find_shape(graph, x)
        if x_shape is not None:
            tensor_shapes[y] = tuple(int(v) for v in x_shape)

        t_out = ensure_tensor(y, shape=tensor_shapes.get(y), dtype="int8")
        scale, q_scheme, q_axis_val = _scale_to_ir(scale_arr, axis)
        t_out.scale = scale
        t_out.q_scheme = q_scheme
        t_out.q_axis = q_axis_val

        # If DQ annotates initializer weights, annotate the initializer too
        if op_type == "DequantizeLinear" and x in init_map:
            t_in = ensure_tensor(x, shape=tensor_shapes.get(x), dtype=_np_dtype_to_ir(init_map[x].dtype))
            t_in.scale = scale
            t_in.q_scheme = q_scheme
            t_in.q_axis = q_axis_val
            t_in.metadata["param_alias"] = True
            t_out.metadata["param_alias"] = True
            t_out.metadata["param_alias_root"] = x

        if op_type == "QuantizeLinear":
            t_in = ensure_tensor(x, shape=tensor_shapes.get(x), dtype="int8")
            t_in.scale = scale
            t_in.q_scheme = q_scheme
            t_in.q_axis = q_axis_val

        storage_alias[y] = x
        return

    if op_type == "Flatten":
        if len(node.input) != 1 or len(node.output) != 1:
            fail(node, f"Flatten expects 1 input/1 output, got inputs={list(node.input)}, outputs={list(node.output)}")
        x, y = node.input[0], node.output[0]
        axis = int(_get_attr(node, "axis", 1))
        if axis != 1:
            fail(node, f"Flatten axis != 1 not supported (axis={axis})")
        x_shape = tensor_shapes.get(x) or _find_shape(graph, x)
        if x_shape is None:
            fail(node, f"Flatten requires known input shape for '{x}'")
        n, c, h, w = _force_4d(_sanitize_shape(x_shape))
        tensor_shapes[y] = (n, c * h * w, 1, 1)
        t_in = ensure_tensor(x, shape=tensor_shapes.get(x))
        t_out = ensure_tensor(y, shape=tensor_shapes[y], dtype=t_in.dtype)
        t_out.scale = t_in.scale
        t_out.q_scheme = t_in.q_scheme
        t_out.q_axis = t_in.q_axis
        storage_alias[y] = x
        program.add_op(VcFlatten(name=node.name or f"Flatten_{y}", inputs=[x], outputs=[y], axis=axis))
        return

    if op_type == "Reshape":
        if len(node.input) < 2 or len(node.output) != 1:
            fail(node, f"Reshape expects 2 inputs and 1 output, got inputs={list(node.input)}, outputs={list(node.output)}")
        x = node.input[0]
        shape_name = node.input[1]
        y = node.output[0]

        x_shape = tensor_shapes.get(x) or _find_shape(graph, x)
        if x_shape is None:
            fail(node, f"Reshape requires known input shape for '{x}'")

        shape_arr = get_const(shape_name)
        if shape_arr is None:
            fail(node, f"Reshape shape input must be initializer/constant, got '{shape_name}'")
        target = [int(v) for v in np.asarray(shape_arr).flatten().tolist()]

        out_shape = _apply_reshape(tuple(int(v) for v in x_shape), target, fail=lambda m: fail(node, m))
        tensor_shapes[y] = out_shape
        t_in = ensure_tensor(x, shape=tensor_shapes.get(x))
        t_out = ensure_tensor(y, shape=out_shape, dtype=t_in.dtype)
        t_out.scale = t_in.scale
        t_out.q_scheme = t_in.q_scheme
        t_out.q_axis = t_in.q_axis
        storage_alias[y] = x
        program.add_op(VcReshape(name=node.name or f"Reshape_{y}", inputs=[x], outputs=[y], new_shape=out_shape))
        return

    fail(node, f"Internal error: unhandled view node type '{op_type}'")


# -----------------------------------------------------------------------------
# Conv / Pool / FC handlers
# -----------------------------------------------------------------------------

def _handle_conv(
    graph,
    node,
    node_id: str,
    ensure_tensor,
    tensor_shapes: Dict[str, Tuple[int, ...]],
    init_map: Dict[str, np.ndarray],
    const_map: Dict[str, np.ndarray],
    storage_alias: Dict[str, str],
    program: VcProgram,
    resolve_storage,
    fail,
) -> None:
    if len(node.input) < 2 or len(node.output) != 1:
        fail(node, f"Conv expects >=2 inputs and 1 output, got inputs={list(node.input)}, outputs={list(node.output)}")

    x_name = resolve_storage(node.input[0])
    w_name = node.input[1]
    b_name = node.input[2] if len(node.input) >= 3 and node.input[2] else None
    y_name = resolve_storage(node.output[0])

    auto_pad = _get_attr(node, "auto_pad", "NOTSET")
    if auto_pad not in ("NOTSET", "", None):
        fail(node, f"Conv auto_pad not supported: {auto_pad}")

    dilations = _get_attr(node, "dilations", [1, 1])
    if list(dilations) != [1, 1]:
        fail(node, f"Conv dilation not supported: {dilations}")

    strides = _get_attr(node, "strides", [1, 1])
    pads = _get_attr(node, "pads", [0, 0, 0, 0])
    group = int(_get_attr(node, "group", 1))

    pad_top, pad_left, pad_bottom, pad_right = _pads_to_tb_lr(pads)
    if not isinstance(strides, list) or len(strides) != 2:
        fail(node, f"Conv strides must be a 2D list, got {strides!r}")
    sh, sw = int(strides[0]), int(strides[1])
    if sh not in (1, 2) or sw not in (1, 2):
        fail(node, f"Conv stride not supported by ISA: strides={strides} (only 1 or 2 supported).")
    if pad_top not in (0, 1) or pad_bottom not in (0, 1) or pad_left not in (0, 1) or pad_right not in (0, 1):
        fail(
            node,
            "Conv padding not supported by ISA: "
            f"(pad_top,pad_bottom,pad_left,pad_right)=({pad_top},{pad_bottom},{pad_left},{pad_right}) "
            "(each must be 0 or 1).",
        )

    x = ensure_tensor(x_name)

    # Resolve weight/bias to initializer-root names for backend friendliness
    w_root = _resolve_to_initializer(w_name, init_map, storage_alias)
    if w_root is None:
        fail(node, f"Weight '{w_name}' is not an initializer (or alias to initializer); dynamic weights are not supported.")
    _validate_weight_int8(w_root, init_map, fail, node)

    w_shape = _get_tensor_shape_or_fail(w_root, tensor_shapes, init_map, const_map, storage_alias, fail, node)
    if len(w_shape) < 4:
        fail(node, f"Conv weight must be 4D (O,I,Kh,Kw), got shape={w_shape} for '{w_root}'")

    cout = int(w_shape[0])
    kh = int(w_shape[2])
    kw = int(w_shape[3])

    _, cin, h_in, w_in = x.shape

    if group == cin and group > 1 and cout != cin:
        fail(node, f"Depthwise multiplier>1 not supported: cin={cin}, cout={cout}, group={group}, weight='{w_root}'")

    h_out = (h_in + pad_top + pad_bottom - kh) // sh + 1
    w_out = (w_in + pad_left + pad_right - kw) // sw + 1
    if h_out <= 0 or w_out <= 0:
        fail(node, f"Conv produced non-positive output shape: H_out={h_out}, W_out={w_out}")

    tensor_shapes[y_name] = (x.shape[0], cout, h_out, w_out)
    y = ensure_tensor(y_name, shape=tensor_shapes[y_name])

    w = ensure_tensor(w_root, shape=w_shape, dtype=_np_dtype_to_ir(init_map[w_root].dtype))
    bias_root = None
    if b_name:
        bias_root = _resolve_to_initializer(b_name, init_map, storage_alias)
        if bias_root is None:
            fail(node, f"Bias '{b_name}' is not an initializer (or alias to initializer); dynamic bias not supported.")
        _validate_bias_dtype(bias_root, init_map, fail, node)
        ensure_tensor(bias_root, dtype=_np_dtype_to_ir(init_map[bias_root].dtype))

    kernel = (kh, kw)
    stride = (sh, sw)
    padding = (pad_top, pad_bottom, pad_left, pad_right)

    if group == cin == cout and kh == 3 and kw == 3:
        op = VcDepthwiseConv(
            name=node_id,
            inputs=[x.name],
            outputs=[y.name],
            weight=w.name,
            bias=bias_root,
            activation=None,
            kernel=kernel,
            stride=stride,
            padding=padding,
            groups=group,
        )
    elif kh == 1 and kw == 1 and group == 1:
        op = VcPointwiseConv(
            name=node_id,
            inputs=[x.name],
            outputs=[y.name],
            weight=w.name,
            bias=bias_root,
            activation=None,
            stride=stride,
            padding=padding,
        )
    elif kh == 3 and kw == 3 and group == 1:
        op = VcConv2D(
            name=node_id,
            inputs=[x.name],
            outputs=[y.name],
            weight=w.name,
            bias=bias_root,
            activation=None,
            kernel=kernel,
            stride=stride,
            padding=padding,
            groups=group,
        )
    else:
        fail(node, f"Unsupported Conv configuration: group={group}, k=({kh},{kw}), stride={strides}, pads={pads}")

    program.add_op(op)


def _handle_avgpool(
    graph,
    node,
    node_id: str,
    ensure_tensor,
    tensor_shapes: Dict[str, Tuple[int, ...]],
    program: VcProgram,
    resolve_storage,
    fail,
) -> None:
    if len(node.input) != 1 or len(node.output) != 1:
        fail(node, f"{node.op_type} expects 1 input/1 output, got inputs={list(node.input)}, outputs={list(node.output)}")
    x_name = resolve_storage(node.input[0])
    y_name = resolve_storage(node.output[0])
    x = ensure_tensor(x_name)

    auto_pad = _get_attr(node, "auto_pad", "NOTSET")
    if auto_pad not in ("NOTSET", "", None):
        fail(node, f"{node.op_type} auto_pad not supported: {auto_pad}")

    ceil_mode = int(_get_attr(node, "ceil_mode", 0))
    if ceil_mode != 0:
        fail(node, f"{node.op_type} ceil_mode not supported: {ceil_mode}")

    if node.op_type == "GlobalAveragePool":
        _, _, h_in, w_in = x.shape
        if h_in != w_in:
            fail(
                node,
                f"GlobalAveragePool only supports square inputs for NPU lowering: got HxW={h_in}x{w_in}.",
            )

        def _is_pow2(v: int) -> bool:
            return v > 0 and (v & (v - 1)) == 0

        def _next_pow2(v: int) -> int:
            p = 1
            while p < v:
                p <<= 1
            return p

        # Preferred lowering:
        # 1) If H=W is power-of-two, lower to repeated 2x2 stride-2 avgpool (e.g. 8x8 -> 4x4 -> 2x2 -> 1x1).
        # 2) If H=W is (2^n - 1), we can pad bottom/right by 1 (ISA supports pad<=1) to reach 2^n, then do (1).
        #
        # NOTE: (2) computes avg over the padded area (divide by 2^(2n)), i.e. it is an approximation of true GAP
        # over (2^n-1)^2. This is currently acceptable for hardware-friendly lowering, but will not be bit-exact
        # with a reference float GAP unless the model/QDQ is calibrated for it.
        h = int(h_in)
        p = h if _is_pow2(h) else _next_pow2(h)
        can_pad_to_pow2 = (p - h) == 1

        if _is_pow2(h) or can_pad_to_pow2:
            cur_name = x_name
            cur_shape = _force_4d(_sanitize_shape(x.shape))
            stages = 0
            t = int(p)
            while t > 1:
                stages += 1
                t //= 2

            for si in range(stages):
                n, c, h_cur, w_cur = cur_shape
                out_shape = (n, c, h_cur // 2, w_cur // 2)
                out_name = y_name if si == (stages - 1) else f"{node_id}_gap_{si}"

                # First stage: optional pad to power-of-two if input was (2^n - 1).
                pad = (0, 0, 0, 0)
                if si == 0 and can_pad_to_pow2:
                    pad = (0, 1, 0, 1)

                t_out = ensure_tensor(out_name, shape=out_shape, dtype=x.dtype)
                # Preserve quant metadata across GAP lowering (view-like transform).
                t_out.scale = getattr(x, "scale", None)
                t_out.q_scheme = getattr(x, "q_scheme", "none")
                t_out.q_axis = getattr(x, "q_axis", None)
                tensor_shapes[out_name] = out_shape

                program.add_op(
                    VcAvgPool(
                        name=f"{node_id}_gap{si}",
                        inputs=[cur_name],
                        outputs=[out_name],
                        kernel=(2, 2),
                        stride=(2, 2),
                        padding=pad,
                        activation=None,
                    )
                )
                cur_name = out_name
                cur_shape = out_shape
            return

        # Fallback: require explicit AveragePool lowering rules (will likely fail in backend if not 2x2).
        kh, kw = h_in, w_in
        sh, sw = 1, 1
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    else:
        kernel_shape = _get_attr(node, "kernel_shape", None)
        if kernel_shape is None or len(kernel_shape) != 2:
            fail(node, f"AveragePool requires 2D kernel_shape, got {kernel_shape}")
        kh, kw = int(kernel_shape[0]), int(kernel_shape[1])
        if kh != 2 or kw != 2:
            fail(node, f"AveragePool kernel not supported: {kh}x{kw} (only 2x2 supported).")
        strides = _get_attr(node, "strides", [1, 1])
        if not isinstance(strides, list) or len(strides) != 2:
            fail(node, f"AveragePool strides must be a 2D list, got {strides!r}")
        sh, sw = int(strides[0]), int(strides[1])
        if sh not in (1, 2) or sw not in (1, 2):
            fail(node, f"AveragePool stride not supported by ISA: strides={strides} (only 1 or 2 supported).")
        pads = _get_attr(node, "pads", [0, 0, 0, 0])
        pad_top, pad_left, pad_bottom, pad_right = _pads_to_tb_lr(pads)
        if pad_top not in (0, 1) or pad_bottom not in (0, 1) or pad_left not in (0, 1) or pad_right not in (0, 1):
            fail(
                node,
                "AveragePool padding not supported by ISA: "
                f"(pad_top,pad_bottom,pad_left,pad_right)=({pad_top},{pad_bottom},{pad_left},{pad_right}) "
                "(each must be 0 or 1).",
            )

    n, c, h_in, w_in = x.shape
    h_out = (h_in + pad_top + pad_bottom - kh) // sh + 1
    w_out = (w_in + pad_left + pad_right - kw) // sw + 1
    if h_out <= 0 or w_out <= 0:
        fail(node, f"{node.op_type} produced non-positive output shape: H_out={h_out}, W_out={w_out}")

    tensor_shapes[y_name] = (n, c, h_out, w_out)
    y = ensure_tensor(y_name, shape=tensor_shapes[y_name])

    op = VcAvgPool(
        name=node_id,
        inputs=[x.name],
        outputs=[y.name],
        kernel=(kh, kw),
        stride=(sh, sw),
        padding=(pad_top, pad_bottom, pad_left, pad_right),
        activation=None,
    )
    program.add_op(op)


def _handle_maxpool(
    graph,
    node,
    node_id: str,
    ensure_tensor,
    tensor_shapes: Dict[str, Tuple[int, ...]],
    program: VcProgram,
    resolve_storage,
    fail,
) -> None:
    if len(node.input) != 1 or len(node.output) != 1:
        fail(node, f"{node.op_type} expects 1 input/1 output, got inputs={list(node.input)}, outputs={list(node.output)}")
    x_name = resolve_storage(node.input[0])
    y_name = resolve_storage(node.output[0])
    x = ensure_tensor(x_name)

    auto_pad = _get_attr(node, "auto_pad", "NOTSET")
    if auto_pad not in ("NOTSET", "", None):
        fail(node, f"{node.op_type} auto_pad not supported: {auto_pad}")

    ceil_mode = int(_get_attr(node, "ceil_mode", 0))
    if ceil_mode != 0:
        fail(node, f"{node.op_type} ceil_mode not supported: {ceil_mode}")

    dilations = _get_attr(node, "dilations", [1, 1])
    if not isinstance(dilations, list) or len(dilations) != 2:
        fail(node, f"{node.op_type} dilations must be a 2D list, got {dilations!r}")
    if int(dilations[0]) != 1 or int(dilations[1]) != 1:
        fail(node, f"{node.op_type} dilations not supported: {dilations} (expect [1,1]).")

    storage_order = int(_get_attr(node, "storage_order", 0))
    if storage_order != 0:
        fail(node, f"{node.op_type} storage_order not supported: {storage_order} (expect 0).")

    if node.op_type == "GlobalMaxPool":
        fail(node, "GlobalMaxPool not supported (only MaxPool 2x2 stride2 supported).")

    kernel_shape = _get_attr(node, "kernel_shape", None)
    if kernel_shape is None or len(kernel_shape) != 2:
        fail(node, f"MaxPool requires 2D kernel_shape, got {kernel_shape}")
    kh, kw = int(kernel_shape[0]), int(kernel_shape[1])
    if kh != 2 or kw != 2:
        fail(node, f"MaxPool kernel not supported: {kh}x{kw} (only 2x2 supported).")
    strides = _get_attr(node, "strides", [1, 1])
    if not isinstance(strides, list) or len(strides) != 2:
        fail(node, f"MaxPool strides must be a 2D list, got {strides!r}")
    sh, sw = int(strides[0]), int(strides[1])
    if sh != 2 or sw != 2:
        fail(node, f"MaxPool stride not supported: {strides} (only 2 supported).")
    pads = _get_attr(node, "pads", [0, 0, 0, 0])
    pad_top, pad_left, pad_bottom, pad_right = _pads_to_tb_lr(pads)
    if pad_top != 0 or pad_bottom != 0 or pad_left != 0 or pad_right != 0:
        fail(
            node,
            "MaxPool padding not supported by ISA: "
            f"(pad_top,pad_bottom,pad_left,pad_right)=({pad_top},{pad_bottom},{pad_left},{pad_right}) "
            "(all must be 0).",
        )

    n, c, h_in, w_in = x.shape
    h_out = (h_in + pad_top + pad_bottom - kh) // sh + 1
    w_out = (w_in + pad_left + pad_right - kw) // sw + 1
    if h_out <= 0 or w_out <= 0:
        fail(node, f"{node.op_type} produced non-positive output shape: H_out={h_out}, W_out={w_out}")

    tensor_shapes[y_name] = (n, c, h_out, w_out)
    y = ensure_tensor(y_name, shape=tensor_shapes[y_name])

    op = VcMaxPool(
        name=node_id,
        inputs=[x.name],
        outputs=[y.name],
        kernel=(kh, kw),
        stride=(sh, sw),
        padding=(pad_top, pad_bottom, pad_left, pad_right),
        activation=None,
    )
    program.add_op(op)


def _handle_gemm_as_fc(
    graph,
    node,
    node_id: str,
    ensure_tensor,
    tensor_shapes: Dict[str, Tuple[int, ...]],
    init_map: Dict[str, np.ndarray],
    const_map: Dict[str, np.ndarray],
    storage_alias: Dict[str, str],
    program: VcProgram,
    resolve_storage,
    fail,
) -> None:
    if len(node.input) < 2 or len(node.output) != 1:
        fail(node, f"Gemm expects >=2 inputs and 1 output, got inputs={list(node.input)}, outputs={list(node.output)}")
    a_name = resolve_storage(node.input[0])
    b_name = node.input[1]
    c_name = node.input[2] if len(node.input) >= 3 and node.input[2] else None
    y_name = resolve_storage(node.output[0])

    alpha = float(_get_attr(node, "alpha", 1))
    beta = float(_get_attr(node, "beta", 1))
    transA = int(_get_attr(node, "transA", 0))
    transB = int(_get_attr(node, "transB", 1))

    if alpha != 1.0 or beta not in (0.0, 1.0):
        fail(node, f"Gemm alpha/beta unsupported (alpha={alpha}, beta={beta}); require alpha=1 and beta in {{0,1}}.")
    if transA != 0:
        fail(node, f"Gemm transA unsupported (transA={transA}); expected 0.")
    if transB not in (0, 1):
        fail(node, f"Gemm transB unsupported (transB={transB}); expected 0 or 1.")

    b_root = _resolve_to_initializer(b_name, init_map, storage_alias)
    if b_root is None:
        fail(node, f"Gemm weight B '{b_name}' is not an initializer (or alias to initializer); dynamic weights not supported.")
    _validate_weight_int8(b_root, init_map, fail, node)

    b_shape = _get_tensor_shape_or_fail(b_root, tensor_shapes, init_map, const_map, storage_alias, fail, node)
    # Frontends may represent FC weights as:
    #   - 2D: [Cout, Cin]
    #   - 4D: [Cout, Cin, 1, 1] (Conv-like)
    if len(b_shape) == 2:
        cout, cin = int(b_shape[0]), int(b_shape[1])
        b_shape = (cout, cin, 1, 1)
        tensor_shapes[b_root] = b_shape
        # Keep init_map and IR tensor in sync.
        if b_root in init_map:
            try:
                init_map[b_root] = init_map[b_root].reshape(b_shape)
            except Exception:
                pass
        wt = program.tensors.get(b_root)
        if wt is not None:
            wt.shape = b_shape
            try:
                wt.data = init_map[b_root].tolist() if b_root in init_map else wt.data
            except Exception:
                pass
    if len(b_shape) != 4 or int(b_shape[2]) != 1 or int(b_shape[3]) != 1:
        fail(node, f"Gemm weight B must be [Cout,Cin,1,1], got shape={b_shape} for '{b_root}'")

    if transB == 1:
        out_features = int(b_shape[0])
        in_features = int(b_shape[1])
    else:
        out_features = int(b_shape[1])
        in_features = int(b_shape[0])

    # VenusCore requires Cout multiple of 4 (NCHWc4). Pad classifier outputs if needed.
    if out_features % 4 != 0:
        padded = ((out_features + 3) // 4) * 4
        pad = padded - out_features

        # Pad weight tensor [Cout,Cin,1,1] with zero rows.
        wt = program.tensors.get(b_root)
        if wt is None:
            fail(node, f"Gemm weight tensor '{b_root}' missing in IR.")
        wt.shape = (padded, in_features, 1, 1)
        if b_root in init_map:
            try:
                arr = init_map[b_root]
                arr4 = arr.reshape((out_features, in_features, 1, 1))
                import numpy as _np  # type: ignore

                new_arr = _np.zeros((padded, in_features, 1, 1), dtype=arr4.dtype)
                new_arr[:out_features, :, :, :] = arr4
                init_map[b_root] = new_arr
                wt.data = new_arr.tolist()
            except Exception:
                pass

        # Pad per-channel scales if present.
        if getattr(wt, "q_scheme", "none") == "symmetric_per_channel":
            sc = getattr(wt, "scale", None)
            if isinstance(sc, list) and len(sc) == out_features:
                wt.scale = list(sc) + [1.0 for _ in range(pad)]

        # Pad bias tensor if present (shape [1,Cout,1,1] or [Cout]).
        if c_name and beta != 0.0:
            bias_root2 = _resolve_to_initializer(c_name, init_map, storage_alias)
            if bias_root2 is None:
                fail(node, f"Gemm bias C '{c_name}' is not an initializer (or alias to initializer); dynamic bias not supported.")
            bt = program.tensors.get(bias_root2)
            if bt is not None:
                bt.shape = (1, padded, 1, 1)
                if bias_root2 in init_map:
                    try:
                        barr = init_map[bias_root2].reshape((1, out_features, 1, 1))
                        import numpy as _np  # type: ignore

                        new_b = _np.zeros((1, padded, 1, 1), dtype=barr.dtype)
                        new_b[:, :out_features, :, :] = barr
                        init_map[bias_root2] = new_b
                        bt.data = new_b.tolist()
                    except Exception:
                        pass
                else:
                    # Best-effort: extend nested list if present.
                    try:
                        if isinstance(bt.data, list) and bt.data and isinstance(bt.data[0], list):
                            row = bt.data[0]
                            while len(row) < padded:
                                row.append([[0]])
                    except Exception:
                        pass

        out_features = padded
        b_shape = (out_features, in_features, 1, 1)
        tensor_shapes[b_root] = b_shape

    a_shape = tensor_shapes.get(a_name) or _find_shape(graph, a_name)
    if a_shape is None:
        fail(node, f"Gemm requires known input A shape for '{a_name}'")

    a_shape4 = _force_4d(_sanitize_shape(a_shape))
    n, c, h, w = a_shape4
    a_features = c * h * w
    if a_features != in_features:
        fail(node, f"Gemm in_features mismatch: A features={a_features} from {a_shape4}, but B expects {in_features} from {b_shape} (transB={transB})")

    tensor_shapes[y_name] = (n, out_features, 1, 1)
    y = ensure_tensor(y_name, shape=tensor_shapes[y_name])

    a = ensure_tensor(a_name, shape=a_shape4)
    b = ensure_tensor(b_root, shape=b_shape, dtype=_np_dtype_to_ir(init_map[b_root].dtype))

    bias_root = None
    if c_name and beta != 0.0:
        bias_root = _resolve_to_initializer(c_name, init_map, storage_alias)
        if bias_root is None:
            fail(node, f"Gemm bias C '{c_name}' is not an initializer (or alias to initializer); dynamic bias not supported.")
        _validate_bias_dtype(bias_root, init_map, fail, node)
        ensure_tensor(bias_root, dtype=_np_dtype_to_ir(init_map[bias_root].dtype))

    op = VcFullyConnected(
        name=node_id,
        inputs=[a.name],
        outputs=[y.name],
        weight=b.name,
        bias=bias_root,
        activation=None,
    )
    program.add_op(op)


# -----------------------------------------------------------------------------
# Connectivity check
# -----------------------------------------------------------------------------

def _check_connectivity(program: VcProgram, graph_inputs: set[str], init_names: set[str], resolve_storage) -> None:
    produced = {out for op in program.ops for out in op.outputs}
    missing: List[str] = []

    def check_ref(ref: str) -> None:
        root = resolve_storage(ref)
        if root in produced or root in graph_inputs or root in init_names:
            return
        missing.append(f"{ref} -> root:{root}")

    for op in program.ops:
        for inp in op.inputs:
            check_ref(inp)
        if getattr(op, "weight", None):
            check_ref(op.weight)  # type: ignore[arg-type]
        if getattr(op, "bias", None):
            check_ref(op.bias)  # type: ignore[arg-type]

    if missing:
        raise ValueError(
            "IR connectivity check failed: tensors referenced by ops do not resolve "
            "to any producer / graph input / initializer:\n  - " + "\n  - ".join(missing)
        )


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------

def _to_numpy(init) -> np.ndarray:
    return onnx.numpy_helper.to_array(init)


def _collect_constant_nodes(graph) -> Dict[str, np.ndarray]:
    const_map: Dict[str, np.ndarray] = {}
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.type == AttributeProto.TENSOR:
                try:
                    const_map[node.output[0]] = onnx.numpy_helper.to_array(attr.t)
                except Exception:
                    pass
    return const_map


def _collect_qdq_info(graph, init_map: Dict[str, np.ndarray], const_map: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    q_scales: Dict[str, np.ndarray] = {}
    q_axis: Dict[str, int] = {}
    dq_scales: Dict[str, np.ndarray] = {}
    dq_axis: Dict[str, int] = {}

    def get_const(name: str) -> Optional[np.ndarray]:
        if name in init_map:
            return init_map[name]
        if name in const_map:
            return const_map[name]
        return None

    for node in graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if len(node.input) < 2 or len(node.output) < 1:
            continue

        x = node.input[0]
        scale_name = node.input[1]
        out = node.output[0]
        axis = int(_get_attr(node, "axis", 0))

        scale_arr = get_const(scale_name)
        if scale_arr is None:
            continue

        if node.op_type == "QuantizeLinear":
            q_scales[out] = scale_arr
            q_axis[out] = axis
            if x in init_map:
                q_scales[x] = scale_arr
                q_axis[x] = axis
        else:
            dq_scales[out] = scale_arr
            dq_axis[out] = axis
            if x in init_map:
                dq_scales[x] = scale_arr
                dq_axis[x] = axis

    return {"q_scales": q_scales, "q_axis": q_axis, "dq_scales": dq_scales, "dq_axis": dq_axis}


def _get_attr(node, key: str, default):
    for attr in getattr(node, "attribute", []):
        if attr.name != key:
            continue
        if attr.type == AttributeProto.INT:
            return int(attr.i)
        if attr.type == AttributeProto.INTS:
            return [int(x) for x in attr.ints]
        if attr.type == AttributeProto.FLOAT:
            return float(attr.f)
        if attr.type == AttributeProto.FLOATS:
            return [float(x) for x in attr.floats]
        if attr.type == AttributeProto.STRING:
            try:
                return attr.s.decode("utf-8")
            except Exception:
                return default
        try:
            return getattr(attr, "i", default)
        except Exception:
            return default
    return default


def _get_shape_from_value(value) -> Tuple[int, ...] | None:
    try:
        dims = value.type.tensor_type.shape.dim
        if not dims:
            return None
        out = []
        for d in dims:
            v = int(getattr(d, "dim_value", 0) or 0)
            out.append(max(1, v))
        return tuple(out)
    except Exception:
        return None


def _find_shape(graph, name: str) -> Tuple[int, ...] | None:
    for v in list(graph.value_info) + list(graph.input) + list(graph.output):
        if v.name == name:
            return _get_shape_from_value(v)
    return None


def _sanitize_shape(shape: Tuple[int, ...] | Any) -> Tuple[int, ...]:
    try:
        shp = tuple(int(x) for x in shape)
    except Exception:
        shp = (1, 1, 1, 1)
    return tuple(max(1, int(x)) for x in shp)


def _force_4d(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    if len(shape) == 4:
        return (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
    if len(shape) == 2:
        return (int(shape[0]), int(shape[1]), 1, 1)
    if len(shape) == 3:
        return (int(shape[0]), int(shape[1]), int(shape[2]), 1)
    if len(shape) == 1:
        return (1, int(shape[0]), 1, 1)
    if len(shape) == 0:
        return (1, 1, 1, 1)
    n, c, h = int(shape[0]), int(shape[1]), int(shape[2])
    w = 1
    for v in shape[3:]:
        w *= int(v)
    return (n, c, h, w)


def _pads_to_tb_lr(pads: List[int]) -> Tuple[int, int, int, int]:
    if len(pads) != 4:
        return (0, 0, 0, 0)
    return (int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3]))


def _scale_to_ir(scale_arr: np.ndarray, axis: int) -> Tuple[Any, str, Optional[int]]:
    scale_arr = np.asarray(scale_arr)
    if scale_arr.size == 1:
        return float(scale_arr.reshape(())), "symmetric_per_tensor", None
    return [float(x) for x in scale_arr.flatten().tolist()], "symmetric_per_channel", int(axis)


def _np_dtype_to_ir(dt) -> str:
    try:
        if np.issubdtype(dt, np.integer):
            if dt == np.int8:
                return "int8"
            if dt == np.uint8:
                return "uint8"
            if dt == np.int32:
                return "int32"
            if dt == np.int16:
                return "int16"
            return f"int{np.dtype(dt).itemsize * 8}"
        if np.issubdtype(dt, np.floating):
            if dt == np.float32:
                return "fp32"
            if dt == np.float16:
                return "fp16"
            return "fp"
        return str(dt)
    except Exception:
        return "unknown"


def _resolve_to_initializer(name: str, init_map: Dict[str, np.ndarray], storage_alias: Dict[str, str]) -> Optional[str]:
    """Return initializer name for `name` or its alias chain."""
    if name in init_map:
        return name
    cur = name
    seen = set()
    while cur in storage_alias and cur not in seen:
        seen.add(cur)
        cur = storage_alias[cur]
        if cur in init_map:
            return cur
    return None


def _get_tensor_shape_or_fail(
    name: str,
    tensor_shapes: Dict[str, Tuple[int, ...]],
    init_map: Dict[str, np.ndarray],
    const_map: Dict[str, np.ndarray],
    storage_alias: Dict[str, str],
    fail,
    node,
) -> Tuple[int, ...]:
    if name in tensor_shapes:
        return _sanitize_shape(tensor_shapes[name])
    if name in init_map:
        return _sanitize_shape(init_map[name].shape)
    if name in const_map:
        return _sanitize_shape(const_map[name].shape)

    cur = name
    seen = set()
    while cur in storage_alias and cur not in seen:
        seen.add(cur)
        cur = storage_alias[cur]
        if cur in tensor_shapes:
            return _sanitize_shape(tensor_shapes[cur])
        if cur in init_map:
            return _sanitize_shape(init_map[cur].shape)
        if cur in const_map:
            return _sanitize_shape(const_map[cur].shape)

    fail(node, f"Cannot resolve shape for tensor '{name}'")
    return (1, 1, 1, 1)


def _validate_weight_int8(name: str, init_map: Dict[str, np.ndarray], fail, node) -> None:
    arr = init_map.get(name, None)
    if arr is None:
        fail(node, f"Weight '{name}' is not an initializer; dynamic weights not supported.")
    if np.issubdtype(arr.dtype, np.floating):
        fail(node, f"Weight '{name}' is floating-point ({arr.dtype}); quantized integer weights required.")
    # Stronger constraint (recommended for your INT8 pipeline)
    if arr.dtype not in (np.int8, np.uint8):
        fail(node, f"Weight '{name}' dtype '{arr.dtype}' not supported; expected int8/uint8.")


def _validate_bias_dtype(name: str, init_map: Dict[str, np.ndarray], fail, node) -> None:
    arr = init_map.get(name, None)
    if arr is None:
        fail(node, f"Bias '{name}' is not an initializer; dynamic bias not supported.")
    if np.issubdtype(arr.dtype, np.floating):
        fail(node, f"Bias '{name}' is floating-point ({arr.dtype}); int32 bias required.")
    if arr.dtype not in (np.int32,):
        fail(node, f"Bias '{name}' dtype '{arr.dtype}' not supported; expected int32.")


def _apply_reshape(in_shape: Tuple[int, ...], target: List[int], fail: Callable[[str], None]) -> Tuple[int, int, int, int]:
    in_shape_s = _sanitize_shape(in_shape)
    in_elems = 1
    for v in in_shape_s:
        in_elems *= int(v)

    neg1_count = sum(1 for v in target if v == -1)
    if neg1_count > 1:
        fail(f"Reshape target has multiple -1: {target}")

    known = 1
    for v in target:
        if v not in (-1, 0):
            known *= int(v)

    out: List[int] = []
    for v in target:
        if v == 0:
            idx = len(out)
            if idx >= len(in_shape_s):
                fail(f"Reshape target uses 0 at idx={idx} but input rank={len(in_shape_s)}")
            out.append(int(in_shape_s[idx]))
        elif v == -1:
            if known == 0:
                fail(f"Reshape cannot infer -1 with known product 0, target={target}")
            out.append(int(in_elems // known))
        else:
            out.append(int(v))

    out_elems = 1
    for v in out:
        out_elems *= int(v)
    if out_elems != in_elems:
        fail(f"Reshape element count mismatch: in={in_shape_s} ({in_elems}) -> out={out} ({out_elems})")

    return _force_4d(tuple(out))


def _maybe_fuse_relu(program: VcProgram, relu_in: str, relu_out: str, resolve_storage) -> bool:
    """
    Attempt to fuse Relu into the last op. Success requires:
      - last_op.outputs contains relu_in, or resolve_storage(relu_in) matches resolve_storage(out)
      - last_op has an activation field
    On success, rewires output name to relu_out and sets activation="relu".
    """
    return _maybe_fuse_activation(program, relu_in, relu_out, resolve_storage, act="relu")


def _maybe_fuse_activation(program: VcProgram, act_in: str, act_out: str, resolve_storage, *, act: str) -> bool:
    """
    Attempt to fuse an activation-like op into the last compute op.

    Success requires:
      - last_op.outputs contains act_in, or resolve_storage(act_in) matches resolve_storage(out)
      - last_op has an activation field

    On success, rewires the output name to act_out and sets last_op.activation=act.
    """
    if not program.ops:
        return False
    last_op = program.ops[-1]
    outs = list(getattr(last_op, "outputs", []))
    act_root = resolve_storage(act_in)
    matched = None
    for o in outs:
        if o == act_in or resolve_storage(o) == act_root:
            matched = o
            break
    if matched is None:
        return False
    if not hasattr(last_op, "activation"):
        return False
    try:
        last_op.activation = act
    except Exception:
        return False
    last_op.outputs = [act_out if t == matched else t for t in outs]  # type: ignore[assignment]
    return True


def _get_clip_min_max(node, init_map: Dict[str, np.ndarray], const_map: Dict[str, np.ndarray], fail) -> tuple[float, float]:
    """
    Extract Clip(min,max) bounds as Python floats.

    Supported forms (strict):
      - Opset >= 11: Clip(data, min, max) where min/max are initializer/Constant.

    Attribute-based Clip is intentionally not supported to avoid ambiguity.
    """
    # Prefer input-based bounds (Clip-11 style).
    if len(node.input) >= 3 and node.input[1] and node.input[2]:
        min_name = node.input[1]
        max_name = node.input[2]
        if min_name not in init_map and min_name not in const_map:
            fail(node, f"Clip min must be initializer/constant, got '{min_name}'")
        if max_name not in init_map and max_name not in const_map:
            fail(node, f"Clip max must be initializer/constant, got '{max_name}'")
        min_arr = init_map.get(min_name) if min_name in init_map else const_map[min_name]
        max_arr = init_map.get(max_name) if max_name in init_map else const_map[max_name]
        try:
            min_v = float(np.array(min_arr).reshape(()))
            max_v = float(np.array(max_arr).reshape(()))
        except Exception:
            fail(node, f"Clip min/max must be scalar constants, got min={min_arr.shape}, max={max_arr.shape}")
        return min_v, max_v

    fail(
        node,
        f"Clip requires min/max constant inputs (got inputs={list(node.input)}). "
        "Attribute-based Clip is not supported in this loader.",
    )


def _fail(node, msg: str) -> None:
    prefix = "[ONNX_LOADER][ERROR]"
    if node is None:
        print(f"{prefix} {msg}")
        raise ValueError(msg)

    node_name = getattr(node, "name", "") or "<unnamed>"
    op_type = getattr(node, "op_type", "<unknown>")
    inputs = list(getattr(node, "input", []))
    outputs = list(getattr(node, "output", []))

    print(f"{prefix} node='{node_name}' op='{op_type}' inputs={inputs} outputs={outputs}")
    print(f"{prefix} {msg}")
    raise ValueError(f"{op_type}({node_name}): {msg}")
