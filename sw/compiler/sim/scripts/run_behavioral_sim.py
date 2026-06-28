# -*- coding: utf-8 -*-
"""
Run the VenusCore functional simulator on a compiled artifact directory.

This is a lightweight helper for validating uops.bin/params.bin behavior against
optional NumPy test vectors.

Inputs:
  - A compiler output directory containing:
      uops.bin, params.bin, metadata.json
    Optionally:
      debug_ir.json (used to infer input/output scales and output shape)

Optional:
  - A float32/float64 input feature .npy file in NCHW shape (N=1).
    The script will symmetric-quantize it using the inferred input scale and
    pack to NCHWc4 bytes before running the uOPs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, List, Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run VenusCore behavioral simulator on compiled artifacts.")
    p.add_argument("--out-dir", type=Path, required=True, help="Compiler output directory (contains uops.bin/params.bin).")
    default_input = Path("sim/testvectors/testvectors_kws/input_f32.npy")
    default_expected = Path("sim/testvectors/testvectors_kws/qdq_logits.npy")
    p.add_argument(
        "--input-npy",
        type=Path,
        default=default_input if default_input.exists() else None,
        help="Optional float input NPY (NCHW, N=1). Defaults to sim/testvectors/input_f32.npy if present.",
    )
    p.add_argument(
        "--expected-npy",
        type=Path,
        default=default_expected if default_expected.exists() else None,
        help="Optional expected float output NPY for comparison. Defaults to sim/testvectors/qdq_logits.npy if present.",
    )
    p.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Optional ONNX model path for per-layer alignment.",
    )
    p.add_argument(
        "--per-layer",
        action="store_true",
        help="If set, compare each compiled layer OFM against ONNXRuntime intermediate outputs.",
    )
    p.add_argument(
        "--dump-layer-npy",
        type=Path,
        default=None,
        help="Optional directory to dump per-layer SIM/ONNX outputs as .npy files.",
    )
    p.add_argument(
        "--mismatch-threshold",
        type=float,
        default=1.0,
        help="Threshold to report the first large per-layer mismatch (default: 1.0 in float domain).",
    )
    p.add_argument("--input-scale", type=float, default=None, help="Override input tensor scale (symmetric).")
    p.add_argument("--output-scale", type=float, default=None, help="Override output tensor scale (symmetric).")
    p.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-K to report in classification summary (default: 5).",
    )
    p.add_argument(
        "--batch-list",
        type=Path,
        default=None,
        help=(
            "Optional batch manifest file. Each non-empty line is either "
            "`input.npy` or `input.npy,expected.npy`. "
            "When set, the script runs all samples and prints an aggregate summary."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Verbose uOP execution logging.")
    return p.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_io_from_debug_ir(debug_ir: dict) -> Tuple[float | None, float | None, Tuple[int, int, int, int] | None]:
    """
    Infer input/output scales and final output shape from debug_ir.json.

    Returns:
      (input_scale, output_scale, output_shape_nchw)
    """
    ops = debug_ir.get("ops", [])
    tensors = debug_ir.get("tensors", {})
    if not ops:
        return None, None, None

    first = ops[0]
    last = ops[-1]
    input_name = first.get("inputs", [None])[0]
    output_name = last.get("outputs", [None])[0]

    in_scale = None
    out_scale = None
    out_shape = None
    if input_name and input_name in tensors:
        in_scale = tensors[input_name].get("scale")
    if output_name and output_name in tensors:
        out_scale = tensors[output_name].get("scale")
        shape = tensors[output_name].get("shape")
        if isinstance(shape, list) and len(shape) == 4:
            out_shape = tuple(int(x) for x in shape)  # type: ignore[assignment]
    try:
        in_scale = float(in_scale) if in_scale is not None else None
    except Exception:
        in_scale = None
    try:
        out_scale = float(out_scale) if out_scale is not None else None
    except Exception:
        out_scale = None
    return in_scale, out_scale, out_shape


def _try_load_testvector_meta() -> dict | None:
    """
    Best-effort loader for sim/testvectors/meta.json.

    The file is optional; when present, it may contain:
      - label: int
      - commands: list[str] (index -> command string)
    """
    meta_path = Path("sim/testvectors/meta.json")
    if not meta_path.exists():
        return None
    try:
        return _load_json(meta_path)
    except Exception:
        return None


def _load_batch_list(path: Path) -> List[Tuple[Path, Path | None]]:
    """
    Parse a simple batch list file.

    Each non-empty, non-comment line is either:
      - input.npy
      - input.npy,expected.npy

    Returns:
      List of (input_path, expected_path_or_none).
    """
    items: List[Tuple[Path, Path | None]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Allow CSV or whitespace separation.
        if "," in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
        else:
            parts = [p for p in line.split() if p]
        if not parts:
            continue
        if len(parts) == 1:
            items.append((Path(parts[0]), None))
        elif len(parts) == 2:
            items.append((Path(parts[0]), Path(parts[1])))
        else:
            raise ValueError(f"Invalid batch-list line: {raw!r}")
    return items


def _topk_indices(x, k: int) -> List[int]:
    """Return indices of the top-k largest elements of a 1D array-like (descending)."""
    import numpy as np

    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        return []
    k_eff = max(1, min(int(k), int(arr.size)))
    # argsort is fine for small logits; keep it simple and stable.
    idx = np.argsort(arr)[::-1][:k_eff]
    return [int(i) for i in idx.tolist()]


def _margin_top1_top2(x) -> float | None:
    """Return (top1 - top2) margin for a 1D array-like, or None if not enough elements."""
    import numpy as np

    arr = np.asarray(x).reshape(-1)
    if arr.size < 2:
        return None
    # Use partial selection to avoid sorting the entire vector.
    top2 = np.partition(arr, -2)[-2:]
    top1 = float(np.max(top2))
    top2v = float(np.min(top2))
    return top1 - top2v


def _collect_layer_output_names(debug_ir: dict) -> List[str]:
    """Return the ordered list of per-op output tensor names from debug_ir.json."""
    out_names: List[str] = []
    for op in debug_ir.get("ops", []):
        outs = op.get("outputs", [])
        if isinstance(outs, list) and outs:
            name = outs[0]
            if isinstance(name, str) and name:
                out_names.append(name)
    return out_names


def _run_onnx_collect_outputs(
    onnx_path: Path, input_name: str, input_value: "object", output_names: List[str]
) -> Dict[str, "object"]:
    """
    Run ONNXRuntime once and fetch a set of intermediate tensors.

    ORT cannot fetch arbitrary intermediates unless they are graph outputs, so
    we temporarily append requested value names to graph.output.
    """
    try:
        import onnx  # type: ignore
        import onnxruntime as ort  # type: ignore
        from onnx import helper  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Per-layer alignment requires 'onnx' and 'onnxruntime'. "
            "Install with `pip install onnx onnxruntime`."
        ) from exc

    model = onnx.load(str(onnx_path))
    graph = model.graph

    existing_outputs = {o.name for o in graph.output}
    # Add requested outputs if they exist as value names in the graph.
    to_add: List[str] = []
    for name in output_names:
        if name in existing_outputs:
            continue
        # If the name appears in value_info, we can add it as an output.
        if any(v.name == name for v in list(graph.value_info) + list(graph.input)):
            to_add.append(name)
        else:
            # Some tensors may be pruned/renamed by export; skip missing ones.
            continue

    for name in to_add:
        # Type/shape may be unknown; ORT can still materialize many intermediates.
        graph.output.append(helper.make_empty_tensor_value_info(name))

    so = ort.SessionOptions()
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    outs = sess.run(output_names, {input_name: input_value})
    return {k: v for k, v in zip(output_names, outs)}


def _quantize_symmetric_int8(x_f, scale: float) -> "object":
    """Quantize float array to int8 with symmetric scale."""
    import numpy as np

    if scale <= 0.0:
        raise ValueError(f"input_scale must be positive, got {scale}")
    q = np.round(x_f / scale).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q


def _pack_nchw_to_nchwc4_bytes(x_i8) -> bytes:
    """
    Pack int8 NCHW tensor into NCHWc4 bytes.

    Layout: for each (n, c4, h, w) there is a 32-bit word (4 bytes):
      byte0 = ch0, byte1 = ch1, byte2 = ch2, byte3 = ch3
    """
    import numpy as np

    if x_i8.ndim != 4:
        raise ValueError(f"Expected NCHW input, got shape {x_i8.shape}")
    n, c, h, w = x_i8.shape
    if n != 1:
        raise ValueError(f"Only N=1 supported, got N={n}")

    c4 = (c + 3) // 4
    out = np.zeros((n, c4, h, w, 4), dtype=np.uint8)
    for ch in range(c):
        g = ch // 4
        lane = ch % 4
        out[0, g, :, :, lane] = x_i8[0, ch, :, :].view(np.uint8)
    return out.tobytes()


def _unpack_nchwc4_bytes_to_nchw(data: bytes, shape_nchw: Tuple[int, int, int, int]) -> "object":
    """Unpack NCHWc4 bytes back to int8 NCHW tensor (N=1)."""
    import numpy as np

    n, c, h, w = shape_nchw
    if n != 1:
        raise ValueError(f"Only N=1 supported, got N={n}")
    c4 = (c + 3) // 4
    expected = n * c4 * h * w * 4
    if len(data) < expected:
        raise ValueError(f"Not enough OFM bytes: need {expected}, got {len(data)}")
    arr = np.frombuffer(data[:expected], dtype=np.uint8).reshape((n, c4, h, w, 4))
    out = np.zeros((n, c, h, w), dtype=np.int8)
    for ch in range(c):
        g = ch // 4
        lane = ch % 4
        out[0, ch, :, :] = arr[0, g, :, :, lane].view(np.int8)
    return out


def _dump_layer_ofm_int8(uops_layer: List["object"], mem: "object") -> Tuple["object", Tuple[int, int, int, int]]:
    """
    Reconstruct the full-layer OFM int8 tensor (NCHW, N=1) from memory.

    This uses ISA semantics:
      - FO_ADDR encodes the base of the selected C4_OUT group range (no Y offset)
      - Y offset is expressed via Y_INDEX and FO_STRIDE inside the cluster
      - OFM layout is NCHWc4 (packed into u32 words)
    """
    import numpy as np

    if not uops_layer:
        raise ValueError("Empty layer uOP list.")

    w_tile = int(uops_layer[0].w_tile)
    fo_stride = int(uops_layer[0].fo_stride)
    if w_tile <= 0 or fo_stride <= 0 or fo_stride % w_tile != 0:
        raise ValueError(f"Invalid OFM shape from uOP: w_tile={w_tile}, fo_stride={fo_stride}")
    ofm_h = fo_stride // w_tile
    ofm_w = w_tile

    base0 = min(int(u.fo_addr) for u in uops_layer)
    # Determine total C4_OUT by looking at the maximum end group written by any uOP.
    max_c4_end = 0
    for u in uops_layer:
        group_start = (int(u.fo_addr) - base0) // (int(u.fo_stride) * 4)
        max_c4_end = max(max_c4_end, group_start + int(u.c4_out))
    c4_total = int(max_c4_end)
    cout_total = c4_total * 4

    out = np.zeros((1, cout_total, ofm_h, ofm_w), dtype=np.int8)

    for u in uops_layer:
        group_start = (int(u.fo_addr) - base0) // (int(u.fo_stride) * 4)
        for oy in range(int(u.h_tile)):
            gy = int(u.y_index) + oy
            if gy < 0 or gy >= ofm_h:
                continue
            for ox in range(ofm_w):
                for g in range(int(u.c4_out)):
                    word_addr = int(u.fo_addr) + (g * int(u.fo_stride) + gy * ofm_w + ox) * 4
                    w = mem.read_u32(word_addr)
                    b0 = (w >> 0) & 0xFF
                    b1 = (w >> 8) & 0xFF
                    b2 = (w >> 16) & 0xFF
                    b3 = (w >> 24) & 0xFF
                    ch_base = (group_start + g) * 4
                    out[0, ch_base + 0, gy, ox] = np.uint8(b0).view(np.int8)
                    out[0, ch_base + 1, gy, ox] = np.uint8(b1).view(np.int8)
                    out[0, ch_base + 2, gy, ox] = np.uint8(b2).view(np.int8)
                    out[0, ch_base + 3, gy, ox] = np.uint8(b3).view(np.int8)

    return out, (1, cout_total, ofm_h, ofm_w)


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out_dir
    uops_path = out_dir / "uops.bin"
    params_path = out_dir / "params.bin"
    meta_path = out_dir / "metadata.json"

    if not uops_path.exists() or not params_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing uops.bin/params.bin/metadata.json under {out_dir}")

    metadata = _load_json(meta_path)
    meta_core = metadata.get("metadata", metadata)

    debug_ir_path = out_dir / "debug_ir.json"
    debug_ir = _load_json(debug_ir_path) if debug_ir_path.exists() else None

    in_scale, out_scale, out_shape = (None, None, None)
    if debug_ir is not None:
        in_scale, out_scale, out_shape = _infer_io_from_debug_ir(debug_ir)

    if args.input_scale is not None:
        in_scale = args.input_scale
    if args.output_scale is not None:
        out_scale = args.output_scale

    if out_shape is None:
        raise ValueError("Cannot infer output shape; please compile with --dump-debug to produce debug_ir.json.")

    # Lazy imports to keep this script optional.
    from sim.behavioral.venuscore_sim import SparseWordMemory, SimConfig, VenusCoreSim, load_uops

    uops = load_uops(str(uops_path))
    if not uops:
        raise ValueError("No uOPs decoded from uops.bin.")

    mem = SparseWordMemory()
    cfg = SimConfig()
    sim = VenusCoreSim(cfg, mem)

    # Load params.bin at the minimum PARAM_ADDR referenced by uOPs.
    param_base = min(int(u.param_addr) for u in uops)
    params_blob = params_path.read_bytes()
    mem.load_bytes(param_base, params_blob)

    # Batch mode: run multiple input vectors and print aggregate statistics.
    if args.batch_list is not None:
        if args.per_layer:
            raise ValueError("Batch mode does not support --per-layer. Run a single sample for per-layer alignment.")
        if not args.batch_list.exists():
            raise FileNotFoundError(f"--batch-list file not found: {args.batch_list}")
        if in_scale is None:
            raise ValueError("Cannot infer input_scale; pass --input-scale or compile with debug_ir.json.")
        if out_scale is None:
            raise ValueError("Cannot infer output_scale; pass --output-scale or compile with debug_ir.json.")

        import numpy as np

        tv_meta = _try_load_testvector_meta()
        commands: List[str] | None = None
        if isinstance(tv_meta, dict):
            cmds = tv_meta.get("commands")
            if isinstance(cmds, list) and all(isinstance(x, str) for x in cmds):
                commands = cmds

        def _idx_to_name(idx: int) -> str:
            if commands is not None and 0 <= idx < len(commands):
                return commands[idx]
            return str(idx)

        items = _load_batch_list(args.batch_list)
        if not items:
            raise ValueError(f"--batch-list is empty: {args.batch_list}")

        fi_base = min(int(u.fi_addr) for u in uops)
        output_base = int(meta_core.get("output_base", min(int(u.fo_addr) for u in uops)))
        output_size = int(meta_core.get("output_size", 0))
        if output_size <= 0:
            n, c, h, w = out_shape
            c4 = (c + 3) // 4
            output_size = c4 * h * w * 4

        total = 0
        with_expected = 0
        top1_match = 0
        ref_in_sim_topk = 0
        max_abs_diffs: List[float] = []
        sim_margins: List[float] = []
        ref_margins: List[float] = []

        print(f"[SIM] Batch mode: {len(items)} samples (topk={int(args.topk)})")
        for idx, (inp_path, exp_path) in enumerate(items):
            if not inp_path.exists():
                raise FileNotFoundError(f"Batch input not found: {inp_path}")
            x_f = np.load(str(inp_path))
            x_i8 = _quantize_symmetric_int8(x_f, float(in_scale))
            ifm_bytes = _pack_nchw_to_nchwc4_bytes(x_i8)

            # Use a fresh memory per sample to avoid cross-sample carryover.
            mem_i = SparseWordMemory()
            sim_i = VenusCoreSim(cfg, mem_i)
            mem_i.load_bytes(param_base, params_blob)
            mem_i.load_bytes(fi_base, ifm_bytes)
            sim_i.run(uops, verbose=bool(args.verbose))

            ofm_bytes = mem_i.dump_bytes(output_base, output_size)
            y_i8 = _unpack_nchwc4_bytes_to_nchw(ofm_bytes, out_shape).astype(np.int32)
            y_f = (y_i8.astype(np.float32) * float(out_scale)).reshape(-1)

            sim_pred = int(np.argmax(y_f))
            sim_topk = _topk_indices(y_f, int(args.topk))
            sim_margin = _margin_top1_top2(y_f)
            if sim_margin is not None:
                sim_margins.append(float(sim_margin))

            ref_pred: int | None = None
            ref_top1_in_sim_topk: bool | None = None
            diff: float | None = None
            if exp_path is not None:
                if not exp_path.exists():
                    raise FileNotFoundError(f"Batch expected not found: {exp_path}")
                expected = np.load(str(exp_path)).astype(np.float32).reshape(-1)
                ref_pred = int(np.argmax(expected))
                ref_top1_in_sim_topk = ref_pred in sim_topk
                ref_margin = _margin_top1_top2(expected)
                if ref_margin is not None:
                    ref_margins.append(float(ref_margin))
                diff = float(np.max(np.abs(y_f.reshape(expected.shape) - expected)))
                max_abs_diffs.append(diff)

            total += 1
            if ref_pred is not None:
                with_expected += 1
                if sim_pred == ref_pred:
                    top1_match += 1
                if bool(ref_top1_in_sim_topk):
                    ref_in_sim_topk += 1

            if ref_pred is None:
                print(f"[SIM] S{idx:03d} sim_pred={sim_pred}({_idx_to_name(sim_pred)}) input={inp_path}")
            else:
                hit_str = "hit" if ref_top1_in_sim_topk else "miss"
                print(
                    f"[SIM] S{idx:03d} ref_pred={ref_pred}({_idx_to_name(ref_pred)}) "
                    f"sim_pred={sim_pred}({_idx_to_name(sim_pred)}) topk={hit_str} diff={diff}"
                )

        if with_expected > 0:
            acc = top1_match / with_expected
            topk_acc = ref_in_sim_topk / with_expected
            mean_diff = float(sum(max_abs_diffs) / max(1, len(max_abs_diffs)))
            max_diff = float(max(max_abs_diffs)) if max_abs_diffs else 0.0
            print(f"[SIM] Batch summary: n={total}, with_expected={with_expected}")
            print(f"[SIM] top1_match_rate={acc:.4f} ({top1_match}/{with_expected})")
            print(f"[SIM] ref_in_sim_topk_rate={topk_acc:.4f} ({ref_in_sim_topk}/{with_expected})")
            print(f"[SIM] max_abs_diff_mean={mean_diff:.6f} max_abs_diff_max={max_diff:.6f}")
        else:
            print(f"[SIM] Batch summary: n={total} (no expected logits provided)")

        if sim_margins:
            print(f"[SIM] sim_margin_mean={float(sum(sim_margins)/len(sim_margins)):.6f} sim_margin_min={float(min(sim_margins)):.6f}")
        if ref_margins:
            print(f"[SIM] ref_margin_mean={float(sum(ref_margins)/len(ref_margins)):.6f} ref_margin_min={float(min(ref_margins)):.6f}")

        print("[SIM] done.")
        return

    # Load IFM if provided.
    if args.input_npy is not None:
        import numpy as np

        if in_scale is None:
            raise ValueError("Cannot infer input_scale; pass --input-scale or compile with debug_ir.json.")
        x_f = np.load(str(args.input_npy))

        # If input_npy is a batch (N>1), run batch evaluation and print summary.
        if x_f.ndim == 4 and int(x_f.shape[0]) > 1:
            if args.per_layer:
                raise ValueError("Batch input NPY is not supported with --per-layer. Use a single-sample input.")
            if out_scale is None:
                raise ValueError("Cannot infer output_scale; pass --output-scale or compile with debug_ir.json.")

            expected = None
            if args.expected_npy is not None:
                expected = np.load(str(args.expected_npy)).astype(np.float32)
                if expected.ndim < 1 or int(expected.shape[0]) != int(x_f.shape[0]):
                    raise ValueError(
                        f"Expected NPY batch mismatch: input N={int(x_f.shape[0])}, expected shape={expected.shape}."
                    )

            tv_meta = _try_load_testvector_meta()
            commands: List[str] | None = None
            labels: List[int] | None = None
            if isinstance(tv_meta, dict):
                cmds = tv_meta.get("commands")
                if isinstance(cmds, list) and all(isinstance(x, str) for x in cmds):
                    commands = cmds
                samples = tv_meta.get("samples")
                if isinstance(samples, list) and len(samples) == int(x_f.shape[0]):
                    lbls: List[int] = []
                    ok = True
                    for s in samples:
                        if not isinstance(s, dict) or not isinstance(s.get("label"), int):
                            ok = False
                            break
                        lbls.append(int(s["label"]))
                    if ok:
                        labels = lbls

            def _idx_to_name(idx: int) -> str:
                if commands is not None and 0 <= idx < len(commands):
                    return commands[idx]
                return str(idx)

            fi_base = min(int(u.fi_addr) for u in uops)
            output_base = int(meta_core.get("output_base", min(int(u.fo_addr) for u in uops)))
            output_size = int(meta_core.get("output_size", 0))
            if output_size <= 0:
                n, c, h, w = out_shape
                c4 = (c + 3) // 4
                output_size = c4 * h * w * 4

            total = int(x_f.shape[0])
            with_expected = 0
            top1_match = 0
            ref_in_sim_topk = 0
            ref_acc_vs_label = 0
            sim_acc_vs_label = 0
            max_abs_diffs: List[float] = []
            sim_margins: List[float] = []
            ref_margins: List[float] = []

            print(f"[SIM] Batch NPY mode: N={total} (topk={int(args.topk)})")
            for i in range(total):
                x_one = x_f[i : i + 1].astype(np.float32)
                x_i8 = _quantize_symmetric_int8(x_one, float(in_scale))
                ifm_bytes = _pack_nchw_to_nchwc4_bytes(x_i8)

                mem_i = SparseWordMemory()
                sim_i = VenusCoreSim(cfg, mem_i)
                mem_i.load_bytes(param_base, params_blob)
                mem_i.load_bytes(fi_base, ifm_bytes)
                sim_i.run(uops, verbose=bool(args.verbose))

                ofm_bytes = mem_i.dump_bytes(output_base, output_size)
                y_i8 = _unpack_nchwc4_bytes_to_nchw(ofm_bytes, out_shape).astype(np.int32)
                y_f = (y_i8.astype(np.float32) * float(out_scale)).reshape(-1)

                sim_pred = int(np.argmax(y_f))
                sim_topk = _topk_indices(y_f, int(args.topk))
                sim_margin = _margin_top1_top2(y_f)
                if sim_margin is not None:
                    sim_margins.append(float(sim_margin))

                ref_pred: int | None = None
                diff: float | None = None
                if expected is not None:
                    exp_one = expected[i].reshape(-1)
                    ref_pred = int(np.argmax(exp_one))
                    diff = float(np.max(np.abs(y_f.reshape(exp_one.shape) - exp_one)))
                    max_abs_diffs.append(diff)
                    with_expected += 1
                    if sim_pred == ref_pred:
                        top1_match += 1
                    if ref_pred in sim_topk:
                        ref_in_sim_topk += 1
                    ref_margin = _margin_top1_top2(exp_one)
                    if ref_margin is not None:
                        ref_margins.append(float(ref_margin))

                if labels is not None:
                    true = int(labels[i])
                    if ref_pred is not None and ref_pred == true:
                        ref_acc_vs_label += 1
                    if sim_pred == true:
                        sim_acc_vs_label += 1

                # Print mismatches (or everything in --verbose).
                if bool(args.verbose) or (labels is not None and sim_pred != int(labels[i])) or (ref_pred is not None and sim_pred != ref_pred):
                    parts: List[str] = [f"[SIM] S{i:03d} sim_pred={sim_pred}({_idx_to_name(sim_pred)})"]
                    if ref_pred is not None:
                        parts.append(f"ref_pred={ref_pred}({_idx_to_name(ref_pred)})")
                        parts.append(f"diff={diff}")
                    if labels is not None:
                        parts.append(f"label={int(labels[i])}({_idx_to_name(int(labels[i]))})")
                    print(" ".join(parts))

            if with_expected > 0:
                print(f"[SIM] batch_with_expected={with_expected}/{total}")
                print(f"[SIM] top1_match_rate={top1_match/with_expected:.4f} ({top1_match}/{with_expected})")
                print(f"[SIM] ref_in_sim_topk_rate={ref_in_sim_topk/with_expected:.4f} ({ref_in_sim_topk}/{with_expected})")
                print(
                    f"[SIM] max_abs_diff_mean={float(sum(max_abs_diffs)/len(max_abs_diffs)):.6f} "
                    f"max_abs_diff_max={float(max(max_abs_diffs)):.6f}"
                )
            if labels is not None:
                print(f"[SIM] ref_acc_vs_meta_label={ref_acc_vs_label/total:.4f} ({ref_acc_vs_label}/{total})")
                print(f"[SIM] sim_acc_vs_meta_label={sim_acc_vs_label/total:.4f} ({sim_acc_vs_label}/{total})")
            if sim_margins:
                print(
                    f"[SIM] sim_margin_mean={float(sum(sim_margins)/len(sim_margins)):.6f} "
                    f"sim_margin_min={float(min(sim_margins)):.6f}"
                )
            if ref_margins:
                print(
                    f"[SIM] ref_margin_mean={float(sum(ref_margins)/len(ref_margins)):.6f} "
                    f"ref_margin_min={float(min(ref_margins)):.6f}"
                )

            print("[SIM] done.")
            return

        # Single-sample input.
        x_i8 = _quantize_symmetric_int8(x_f, float(in_scale))
        ifm_bytes = _pack_nchw_to_nchwc4_bytes(x_i8)
        fi_base = min(int(u.fi_addr) for u in uops)
        mem.load_bytes(fi_base, ifm_bytes)

    # Optional per-layer alignment against ONNXRuntime intermediates.
    #
    # IMPORTANT: We must snapshot each layer's OFM immediately after the last uOP
    # of that layer executes. If we run the whole program first and only then
    # read memory, ping-pong reuse will overwrite earlier layer outputs and the
    # comparison will be meaningless.
    if args.per_layer:
        if debug_ir is None:
            raise ValueError("Per-layer alignment requires debug_ir.json (compile with --dump-debug).")
        if args.onnx_model is None or not Path(args.onnx_model).exists():
            raise ValueError("Per-layer alignment requires --onnx-model pointing to an existing ONNX file.")
        if args.input_npy is None or not Path(args.input_npy).exists():
            raise ValueError("Per-layer alignment requires --input-npy.")

        import numpy as np
        x_f = np.load(str(args.input_npy)).astype(np.float32)
        if x_f.ndim == 4 and int(x_f.shape[0]) > 1:
            # Use the first sample for per-layer alignment if a batched NPY is provided.
            x_f = x_f[0:1]

        # Determine ONNX input name.
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            raise ImportError("Per-layer alignment requires onnxruntime.") from exc
        sess = ort.InferenceSession(Path(args.onnx_model).read_bytes(), providers=["CPUExecutionProvider"])
        onnx_in_name = sess.get_inputs()[0].name

        layer_out_names = _collect_layer_output_names(debug_ir)
        onnx_outs = _run_onnx_collect_outputs(Path(args.onnx_model), onnx_in_name, x_f, layer_out_names)

        tensors = debug_ir.get("tensors", {})

        print("[SIM] Per-layer alignment (ONNX vs SIM):")
        # uOP 的 pad_* 是按 tile 的边界标志；需要先基于全量 uOP 做一次聚合，
        # 才能在 SIM 里正确反推全局 IFM/OFM 维度。
        sim.prepare(uops)
        cur_layer_uops: List[object] = []
        layer_idx = 0
        first_mismatch: tuple[int, str, float] | None = None
        first_large_mismatch: tuple[int, str, float] | None = None

        dump_dir: Path | None = args.dump_layer_npy
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)

        for u in uops:
            sim.exec_one(u, verbose=bool(args.verbose))
            cur_layer_uops.append(u)

            # sync 表示“分 tile 后，本层最后一个 tile 的最后一条 uOP”，用于层间同步。
            # last_flag 是 stream-level（整个子图最后一条 uOP），不能用于分层。
            if not bool(u.sync):
                continue

            if layer_idx >= len(layer_out_names):
                cur_layer_uops = []
                layer_idx += 1
                continue

            name = layer_out_names[layer_idx]
            ref = onnx_outs.get(name)
            if ref is None:
                print(f"  L{layer_idx:02d} {name}: skipped (ONNX output not available)")
                cur_layer_uops = []
                layer_idx += 1
                continue

            sim_i8, _sim_shape = _dump_layer_ofm_int8(cur_layer_uops, mem)

            # If reference is float, dequantize sim using debug_ir tensor scale.
            if hasattr(ref, "dtype") and str(ref.dtype).startswith("float"):
                scale = None
                if isinstance(tensors, dict) and name in tensors:
                    scale = tensors[name].get("scale")
                if scale is None:
                    print(f"  L{layer_idx:02d} {name}: skipped (missing scale for float compare)")
                    cur_layer_uops = []
                    layer_idx += 1
                    continue
                try:
                    scale_f = float(scale)
                except Exception:
                    print(f"  L{layer_idx:02d} {name}: skipped (invalid scale {scale!r})")
                    cur_layer_uops = []
                    layer_idx += 1
                    continue
                sim_f = sim_i8.astype("float32") * scale_f
                # Broadcast/reshape to ref if possible.
                try:
                    ref_f = ref.astype("float32")
                    if ref_f.shape != sim_f.shape:
                        # Best-effort reshape if total elements match.
                        if ref_f.size == sim_f.size:
                            sim_f = sim_f.reshape(ref_f.shape)
                        else:
                            print(f"  L{layer_idx:02d} {name}: shape mismatch sim{sim_f.shape} ref{ref_f.shape}")
                            cur_layer_uops = []
                            layer_idx += 1
                            continue
                    diff = float(np.max(np.abs(sim_f - ref_f)))
                    print(f"  L{layer_idx:02d} {name}: max_abs_diff={diff}")
                    if first_mismatch is None and diff > 0.0:
                        first_mismatch = (layer_idx, name, diff)
                    if first_large_mismatch is None and diff > float(args.mismatch_threshold):
                        first_large_mismatch = (layer_idx, name, diff)

                    if dump_dir is not None:
                        safe = name.replace("/", "_").replace(":", "_")
                        np.save(str(dump_dir / f"L{layer_idx:02d}_{safe}_sim_f32.npy"), sim_f)
                        np.save(str(dump_dir / f"L{layer_idx:02d}_{safe}_onnx_f32.npy"), ref_f)
                except Exception as exc:
                    print(f"  L{layer_idx:02d} {name}: compare failed ({exc})")
            else:
                # If reference is int8/uint8, compare in integer space.
                try:
                    ref_arr = np.array(ref)
                    if ref_arr.dtype == np.uint8:
                        ref_i8 = ref_arr.view(np.int8)
                    else:
                        ref_i8 = ref_arr.astype(np.int8)
                    if ref_i8.shape != sim_i8.shape:
                        if ref_i8.size == sim_i8.size:
                            sim_cmp = sim_i8.reshape(ref_i8.shape)
                        else:
                            print(f"  L{layer_idx:02d} {name}: shape mismatch sim{sim_i8.shape} ref{ref_i8.shape}")
                            cur_layer_uops = []
                            layer_idx += 1
                            continue
                    else:
                        sim_cmp = sim_i8
                    diff = int(np.max(np.abs(sim_cmp.astype(np.int16) - ref_i8.astype(np.int16))))
                    print(f"  L{layer_idx:02d} {name}: max_abs_diff_i8={diff}")
                    if first_mismatch is None and diff != 0:
                        first_mismatch = (layer_idx, name, float(diff))
                    if first_large_mismatch is None and diff > int(args.mismatch_threshold):
                        first_large_mismatch = (layer_idx, name, float(diff))

                    if dump_dir is not None:
                        safe = name.replace("/", "_").replace(":", "_")
                        np.save(str(dump_dir / f"L{layer_idx:02d}_{safe}_sim_i8.npy"), sim_cmp)
                        np.save(str(dump_dir / f"L{layer_idx:02d}_{safe}_onnx_i8.npy"), ref_i8)
                except Exception as exc:
                    print(f"  L{layer_idx:02d} {name}: int compare failed ({exc})")

            cur_layer_uops = []
            layer_idx += 1

        if first_mismatch is not None:
            li, lname, ldiff = first_mismatch
            print(f"[SIM] first_mismatch: L{li:02d} {lname} diff={ldiff}")
        if first_large_mismatch is not None:
            li, lname, ldiff = first_large_mismatch
            print(f"[SIM] first_large_mismatch(>{args.mismatch_threshold}): L{li:02d} {lname} diff={ldiff}")
    else:
        sim.run(uops, verbose=bool(args.verbose))

    # Read OFM bytes from output_base/output_size in metadata when present.
    output_base = int(meta_core.get("output_base", min(int(u.fo_addr) for u in uops)))
    output_size = int(meta_core.get("output_size", 0))
    if output_size <= 0:
        # Fallback: compute from out_shape and NCHWc4 packing.
        n, c, h, w = out_shape
        c4 = (c + 3) // 4
        output_size = c4 * h * w * 4
    ofm_bytes = mem.dump_bytes(output_base, output_size)

    # Optional compare against expected float outputs.
    if args.expected_npy is not None:
        import numpy as np

        expected = np.load(str(args.expected_npy)).astype(np.float32)
        y_i8 = _unpack_nchwc4_bytes_to_nchw(ofm_bytes, out_shape).astype(np.int32)
        if out_scale is None:
            raise ValueError("Cannot infer output_scale; pass --output-scale or compile with debug_ir.json.")
        y_f = (y_i8.astype(np.float32) * float(out_scale)).reshape(expected.shape)
        diff = np.max(np.abs(y_f - expected))

        # Classification summary (argmax/top-k on logits).
        exp_idx = int(np.argmax(expected))
        sim_idx = int(np.argmax(y_f))
        tv_meta = _try_load_testvector_meta()
        commands = None
        true_label = None
        if isinstance(tv_meta, dict):
            cmds = tv_meta.get("commands")
            if isinstance(cmds, list) and all(isinstance(x, str) for x in cmds):
                commands = cmds
            if isinstance(tv_meta.get("label"), int):
                true_label = int(tv_meta["label"])

        def _idx_to_name(idx: int) -> str:
            if commands is not None and 0 <= idx < len(commands):
                return commands[idx]
            return str(idx)

        exp_name = _idx_to_name(exp_idx)
        sim_name = _idx_to_name(sim_idx)
        print(f"[SIM] reference_pred={exp_idx} ({exp_name})")
        print(f"[SIM] sim_pred      ={sim_idx} ({sim_name})")
        if true_label is not None:
            true_name = _idx_to_name(true_label)
            print(f"[SIM] meta_label    ={true_label} ({true_name})")

        # Report Top-K lists and margins to help diagnose near-misses.
        k = int(args.topk)
        exp_flat = expected.reshape(-1)
        sim_flat = y_f.reshape(-1)
        exp_topk = _topk_indices(exp_flat, k)
        sim_topk = _topk_indices(sim_flat, k)
        exp_margin = _margin_top1_top2(exp_flat)
        sim_margin = _margin_top1_top2(sim_flat)
        if exp_margin is not None:
            print(f"[SIM] reference_margin(top1-top2)={float(exp_margin):.6f}")
        if sim_margin is not None:
            print(f"[SIM] sim_margin(top1-top2)      ={float(sim_margin):.6f}")
        if exp_topk:
            body = ", ".join(f"{i}:{_idx_to_name(i)}={float(exp_flat[i]):.6f}" for i in exp_topk)
            print(f"[SIM] reference_top{k}: {body}")
        if sim_topk:
            body = ", ".join(f"{i}:{_idx_to_name(i)}={float(sim_flat[i]):.6f}" for i in sim_topk)
            print(f"[SIM] sim_top{k}      : {body}")
        print(f"[SIM] max_abs_diff={float(diff)}")

    print("[SIM] done.")


if __name__ == "__main__":
    main()
