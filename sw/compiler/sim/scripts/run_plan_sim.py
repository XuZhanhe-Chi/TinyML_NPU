# -*- coding: utf-8 -*-
"""
Run a Plan (NPU/CPU/ALIAS) using the Python behavioral simulator.

This validates end-to-end correctness for graphs that include CPU fallback ops
such as Add/Concat.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from venuscore_compiler.isa.uop_format import UOP_SIZE_BYTES


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run VenusCore plan simulation on compiled artifacts.")
    p.add_argument("--out-dir", type=Path, required=True, help="Compiler output directory (contains uops.bin/params.bin/metadata.json).")
    p.add_argument("--input-npy", type=Path, required=True, help="Input float32 NPY. Supports NCHW (N=1) or batch NCHW (N>1).")
    p.add_argument("--expected-npy", type=Path, required=True, help="Expected float32 NPY. Supports matching batch dimension.")
    p.add_argument("--max-samples", type=int, default=1, help="Max samples to run from the batch (default: 1).")
    p.add_argument("--start-idx", type=int, default=0, help="Start index in the batch (default: 0).")
    p.add_argument("--input-scale", type=float, default=None, help="Override input scale.")
    p.add_argument("--output-scale", type=float, default=None, help="Override output scale.")
    p.add_argument(
        "--sim-act-base",
        type=lambda s: int(s, 0),
        default=None,
        help="Override simulated activation base address (default: 0 in offset mode).",
    )
    p.add_argument(
        "--sim-param-base",
        type=lambda s: int(s, 0),
        default=None,
        help="Override simulated param base address (default: non-overlapping base in offset mode).",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose step execution logging.")
    p.add_argument(
        "--quiet-per-sample",
        action="store_true",
        help="Suppress per-sample progress lines and only print final summary.",
    )
    p.add_argument(
        "--dump-output-npy",
        type=Path,
        default=None,
        help="Optional path to dump VenusCore-sim output float32 array (same shape as expected_npy).",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_io_from_debug_ir(debug_ir: dict) -> Tuple[float | None, float | None, Tuple[int, int, int, int] | None]:
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


def _quantize_symmetric_i8(x_f32, scale: float):
    import numpy as np

    if scale <= 0:
        raise ValueError(f"Invalid scale: {scale}")
    q = np.round(x_f32 / float(scale)).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q


def _pack_nchw_to_nchwc4_bytes(x_i8) -> bytes:
    import numpy as np

    if x_i8.ndim != 4:
        raise ValueError(f"Expected NCHW int8, got shape={x_i8.shape}")
    n, c, h, w = x_i8.shape
    if n != 1:
        raise ValueError(f"Only N==1 supported, got N={n}")
    c4 = (c + 3) // 4
    out = np.zeros((c4, h, w, 4), dtype=np.uint8)
    x_u8 = x_i8.astype(np.uint8)
    for ch in range(c):
        g = ch // 4
        k = ch % 4
        out[g, :, :, k] = x_u8[0, ch, :, :]
    return out.tobytes(order="C")


def _unpack_nchwc4_bytes_to_nchw(buf: bytes, shape_nchw: Tuple[int, int, int, int]):
    import numpy as np

    n, c, h, w = shape_nchw
    if n != 1:
        raise ValueError(f"Only N==1 supported, got N={n}")
    c4 = (c + 3) // 4
    expect = c4 * h * w * 4
    if len(buf) != expect:
        raise ValueError(f"Buffer size mismatch: got {len(buf)} bytes, expected {expect}")
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((c4, h, w, 4))
    out = np.zeros((1, c, h, w), dtype=np.int8)
    for ch in range(c):
        g = ch // 4
        k = ch % 4
        out[0, ch, :, :] = arr[g, :, :, k].view(np.int8)
    return out


def _sat_add_i8(a: int, b: int) -> int:
    s = int(a) + int(b)
    if s > 127:
        return 127
    if s < -128:
        return -128
    return s


def _cpu_add(mem, a_addr: int, b_addr: int, y_addr: int, size_bytes: int) -> None:
    import numpy as np

    a = mem.dump_bytes(a_addr, size_bytes)
    b = mem.dump_bytes(b_addr, size_bytes)
    a_i8 = np.frombuffer(a, dtype=np.int8).astype(np.int16)
    b_i8 = np.frombuffer(b, dtype=np.int8).astype(np.int16)
    y = np.clip(a_i8 + b_i8, -128, 127).astype(np.int8)
    mem.load_bytes(y_addr, y.view(np.uint8).tobytes())


def _cpu_add_requant(
    mem,
    a_addr: int,
    b_addr: int,
    y_addr: int,
    size_bytes: int,
    a_scale: float,
    b_scale: float,
    y_scale: float,
) -> None:
    import numpy as np

    if not (a_scale > 0.0 and b_scale > 0.0 and y_scale > 0.0):
        raise ValueError(f"Invalid scales for requant add: a={a_scale}, b={b_scale}, y={y_scale}")

    a = mem.dump_bytes(a_addr, size_bytes)
    b = mem.dump_bytes(b_addr, size_bytes)
    a_i8 = np.frombuffer(a, dtype=np.int8).astype(np.float32)
    b_i8 = np.frombuffer(b, dtype=np.int8).astype(np.float32)
    y_f = a_i8 * float(a_scale) + b_i8 * float(b_scale)
    y_q = np.round(y_f / float(y_scale)).astype(np.int32)
    y_q = np.clip(y_q, -128, 127).astype(np.int8)
    mem.load_bytes(y_addr, y_q.view(np.uint8).tobytes())


def _cpu_apply_activation(mem, y_addr: int, size_bytes: int, activation: str, y_scale: float | None) -> None:
    import numpy as np

    act = str(activation or "NONE").strip().upper()
    if act in ("", "NONE"):
        return
    y = np.frombuffer(mem.dump_bytes(y_addr, size_bytes), dtype=np.int8).astype(np.int16)
    if act == "RELU":
        y = np.maximum(y, 0).astype(np.int8)
        mem.load_bytes(y_addr, y.view(np.uint8).tobytes())
        return
    if act == "RELU6":
        if y_scale is None or y_scale <= 0.0:
            raise ValueError(f"RELU6 requires positive output scale, got {y_scale}")
        q6 = int(round(6.0 / float(y_scale)))
        q6 = max(0, min(127, q6))
        y = np.clip(y, 0, q6).astype(np.int8)
        mem.load_bytes(y_addr, y.view(np.uint8).tobytes())
        return
    raise ValueError(f"Unsupported CPU activation: {activation!r}")


def _cpu_concat_c(mem, plan_tensors: Dict[int, dict], inputs: list[int], output: int) -> None:
    # Simple c4-plane copy: output buffer is laid out as [c4][h][w][k].
    out_desc = plan_tensors[output]
    n, c_out, h, w = (int(x) for x in out_desc["shape"])
    if n != 1:
        raise ValueError("concat only supports N==1 in v1")
    c4_out = (c_out + 3) // 4
    plane_bytes = h * w * 4
    out_addr = int(out_desc["offset_bytes"])

    # Zero-fill padded channels.
    mem.load_bytes(out_addr, b"\x00" * (c4_out * plane_bytes))

    out_c4_base = 0
    c_sum = 0
    for tid in inputs:
        td = plan_tensors[tid]
        n_i, c_i, h_i, w_i = (int(x) for x in td["shape"])
        if (n_i, h_i, w_i) != (n, h, w):
            raise ValueError(f"concat spatial mismatch: {td['name']} shape={td['shape']} vs out={out_desc['shape']}")
        c_sum += c_i
        src_addr = int(td["offset_bytes"])
        c4_i = (c_i + 3) // 4
        rem = c_i & 3
        for g in range(c4_i):
            src_plane = mem.dump_bytes(src_addr + g * plane_bytes, plane_bytes)
            mem.load_bytes(out_addr + (out_c4_base + g) * plane_bytes, src_plane)
            if g == (c4_i - 1) and rem != 0:
                # Zero the padded lanes for each spatial word.
                dst_plane_addr = out_addr + (out_c4_base + g) * plane_bytes
                plane = bytearray(mem.dump_bytes(dst_plane_addr, plane_bytes))
                for wi in range(h * w):
                    base = wi * 4
                    for k in range(rem, 4):
                        plane[base + k] = 0
                mem.load_bytes(dst_plane_addr, bytes(plane))
        out_c4_base += c4_i

    if c_sum != c_out:
        raise ValueError(f"concat channel mismatch: sum_in={c_sum}, out_c={c_out}")


def _align_up(v: int, a: int) -> int:
    if a <= 0:
        return v
    return (int(v) + int(a) - 1) & ~(int(a) - 1)


def _choose_non_overlapping_param_base(arena_bytes: int) -> int:
    """
    Pick a simulated param base that will not overlap the activation arena.

    In offset-address mode, uOP FI/FO/PARAM fields are *offsets* relative to
    independent base pointers (act_base / param_base). The behavioral simulator
    uses a single flat address space, so we must choose distinct bases to avoid
    overlapping params and activations in SparseWordMemory.
    """
    # Keep it small-ish (32-bit) but far from typical activation arenas.
    # Also ensure it's above the arena footprint.
    return max(0x0100_0000, _align_up(int(arena_bytes), 0x1000) + 0x1000)


def _relocate_uops(uops, act_base: int, param_base: int):
    return [
        replace(u, param_addr=int(u.param_addr) + int(param_base), fi_addr=int(u.fi_addr) + int(act_base), fo_addr=int(u.fo_addr) + int(act_base))
        for u in uops
    ]


def _run_plan(mem, sim, uops, meta_core: dict, *, address_mode: str, act_base: int, verbose: bool) -> None:
    plan = meta_core.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("metadata.plan missing; compile with emit_plan enabled.")

    tensors = plan.get("tensors", [])
    steps = plan.get("steps", [])
    quant_scales = plan.get("quant_scales", [])
    if not isinstance(tensors, list) or not isinstance(steps, list):
        raise ValueError("plan tensors/steps malformed")
    if not isinstance(quant_scales, list):
        quant_scales = []

    plan_tensors: Dict[int, dict] = {}
    for t in tensors:
        if isinstance(t, dict):
            plan_tensors[int(t["tensor_id"])] = t

    def _get_scale(tid: int) -> float | None:
        td = plan_tensors.get(tid)
        if not isinstance(td, dict):
            return None
        qidx = td.get("quant_index", None)
        if qidx is None:
            return None
        try:
            qi = int(qidx)
        except Exception:
            return None
        if qi < 0 or qi >= len(quant_scales):
            return None
        try:
            s = float(quant_scales[qi])
        except Exception:
            return None
        return s if s > 0.0 else None

    def _taddr(tid: int) -> int:
        td = plan_tensors.get(int(tid), {})
        off = int(td.get("offset_bytes", 0))
        return (int(act_base) + off) if address_mode == "offset" else off

    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        st = str(s.get("type"))
        if verbose:
            print(f"[PLAN] step {i:03d} type={st} inputs={s.get('inputs')} outputs={s.get('outputs')}")
        if st == "NPU":
            off_words = int(s.get("uop_off_words", 0))
            ln_words = int(s.get("uop_words", 0))
            uop_words_per_uop = int(UOP_SIZE_BYTES // 4)
            if off_words % uop_words_per_uop != 0 or ln_words % uop_words_per_uop != 0:
                raise ValueError(
                    f"uop range not multiple of {uop_words_per_uop} words: off={off_words} ln={ln_words}"
                )
            start = off_words // uop_words_per_uop
            cnt = ln_words // uop_words_per_uop
            sim.run(uops[start : start + cnt], verbose=verbose)
            continue
        if st == "ALIAS":
            continue
        if st != "CPU":
            raise ValueError(f"Unknown step type: {st}")
        kernel = str(s.get("kernel"))
        inputs = [int(x) for x in (s.get("inputs") or [])]
        outputs = [int(x) for x in (s.get("outputs") or [])]
        if kernel == "ADD":
            if len(inputs) != 2 or len(outputs) != 1:
                raise ValueError("ADD expects 2 inputs and 1 output")
            a = plan_tensors[inputs[0]]
            b = plan_tensors[inputs[1]]
            y = plan_tensors[outputs[0]]
            sa = _get_scale(inputs[0])
            sb = _get_scale(inputs[1])
            sy = _get_scale(outputs[0])
            activation = str(s.get("activation", "NONE"))
            if sa is not None and sb is not None and sy is not None and (sa != sb or sa != sy):
                _cpu_add_requant(
                    mem,
                    _taddr(inputs[0]),
                    _taddr(inputs[1]),
                    _taddr(outputs[0]),
                    int(y["size_bytes"]),
                    sa,
                    sb,
                    sy,
                )
            else:
                _cpu_add(mem, _taddr(inputs[0]), _taddr(inputs[1]), _taddr(outputs[0]), int(y["size_bytes"]))
            _cpu_apply_activation(mem, _taddr(outputs[0]), int(y["size_bytes"]), activation, sy)
            continue
        if kernel == "CONCAT_C":
            if len(inputs) < 2 or len(outputs) != 1:
                raise ValueError("CONCAT_C expects >=2 inputs and 1 output")
            # _cpu_concat_c expects tensor descriptors with absolute addresses in offset_bytes.
            if address_mode == "offset":
                abs_tensors: Dict[int, dict] = {}
                for tid, td in plan_tensors.items():
                    if isinstance(td, dict):
                        td2 = dict(td)
                        td2["offset_bytes"] = int(act_base) + int(td.get("offset_bytes", 0))
                        abs_tensors[int(tid)] = td2
                _cpu_concat_c(mem, abs_tensors, inputs, outputs[0])
            else:
                _cpu_concat_c(mem, plan_tensors, inputs, outputs[0])
            continue
        raise ValueError(f"Unsupported CPU kernel: {kernel}")


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir
    meta_path = out_dir / "metadata.json"
    uops_path = out_dir / "uops.bin"
    params_path = out_dir / "params.bin"
    debug_ir_path = out_dir / "debug_ir.json"

    if not meta_path.exists() or not uops_path.exists() or not params_path.exists():
        raise FileNotFoundError("Missing uops.bin/params.bin/metadata.json in --out-dir")

    metadata = _load_json(meta_path)
    meta_core = metadata.get("metadata", metadata)

    debug_ir = _load_json(debug_ir_path) if debug_ir_path.exists() else None
    in_scale, out_scale, out_shape = (None, None, None)
    if debug_ir is not None:
        in_scale, out_scale, out_shape = _infer_io_from_debug_ir(debug_ir)
    if args.input_scale is not None:
        in_scale = args.input_scale
    if args.output_scale is not None:
        out_scale = args.output_scale
    if out_shape is None:
        raise ValueError("Cannot infer output shape; compile with --dump-debug to produce debug_ir.json.")
    if in_scale is None or out_scale is None:
        raise ValueError("Cannot infer input/output scale; pass --input-scale/--output-scale.")

    from sim.behavioral.venuscore_sim import SparseWordMemory, SimConfig, VenusCoreSim, load_uops

    uops = load_uops(str(uops_path))
    if not uops:
        raise ValueError("No uOPs decoded from uops.bin.")
    address_mode = str(meta_core.get("address_mode", "absolute"))
    if address_mode not in ("absolute", "offset"):
        raise ValueError(f"Unknown address_mode in metadata: {address_mode!r}")

    # In offset-address mode, avoid overlapping activation/param spaces in the simulator.
    plan = meta_core.get("plan", {})
    arena_bytes = int(plan.get("arena_bytes", 0)) if isinstance(plan, dict) else 0
    act_base = 0 if address_mode == "offset" else 0
    param_base = _choose_non_overlapping_param_base(arena_bytes) if address_mode == "offset" else min(int(u.param_addr) for u in uops)
    if args.sim_act_base is not None:
        act_base = int(args.sim_act_base)
    if args.sim_param_base is not None:
        param_base = int(args.sim_param_base)

    uops = _relocate_uops(uops, act_base=act_base if address_mode == "offset" else 0, param_base=param_base if address_mode == "offset" else 0)

    mem = SparseWordMemory()
    sim = VenusCoreSim(SimConfig(), mem)
    mem.load_bytes(param_base, params_path.read_bytes())

    import numpy as np

    x_all = np.load(str(args.input_npy), mmap_mode="r")
    y_ref = np.load(str(args.expected_npy), mmap_mode="r")

    if x_all.ndim != 4:
        raise ValueError(f"input_npy must be NCHW or batch NCHW, got shape={x_all.shape}")
    if y_ref.ndim < 2:
        raise ValueError(f"expected_npy must have batch dimension, got shape={y_ref.shape}")

    n_total = int(x_all.shape[0])
    start = int(args.start_idx)
    end = min(n_total, start + int(args.max_samples))
    if start < 0 or start >= n_total:
        raise ValueError(f"start_idx out of range: {start} (n_total={n_total})")

    input_base = int(meta_core.get("input_base", min(int(u.fi_addr) for u in uops)))
    output_base = int(meta_core.get("output_base", min(int(u.fo_addr) for u in uops)))
    input_addr = (act_base + input_base) if address_mode == "offset" else input_base
    output_addr = (act_base + output_base) if address_mode == "offset" else output_base
    output_size = int(meta_core.get("output_size", 0))
    if output_size <= 0:
        n, c, h, w = out_shape
        c4 = (c + 3) // 4
        output_size = c4 * h * w * 4

    max_abs = 0.0
    match = 0
    dumped_outputs = []
    for i in range(start, end):
        # Clear memory between samples except params (params are disjoint in address space).
        mem = SparseWordMemory()
        sim = VenusCoreSim(SimConfig(), mem)
        mem.load_bytes(param_base, params_path.read_bytes())

        x = np.asarray(x_all[i : i + 1, :, :, :], dtype=np.float32)  # N==1
        x_q = _quantize_symmetric_i8(x, float(in_scale))
        mem.load_bytes(input_addr, _pack_nchw_to_nchwc4_bytes(x_q))

        _run_plan(mem, sim, uops, meta_core, address_mode=address_mode, act_base=act_base, verbose=bool(args.verbose))

        ofm_bytes = mem.dump_bytes(output_addr, output_size)
        y_i8 = _unpack_nchwc4_bytes_to_nchw(ofm_bytes, out_shape).astype(np.int32)
        y_ref_i = np.asarray(y_ref[i], dtype=np.float32)
        y_f = (y_i8.astype(np.float32) * float(out_scale)).reshape(y_ref_i.shape)
        dumped_outputs.append(y_f.astype(np.float32, copy=False))

        diff = float(np.max(np.abs(y_f - y_ref_i)))
        max_abs = max(max_abs, diff)
        sim_idx = int(np.argmax(y_f.reshape(-1)))
        ref_idx = int(np.argmax(y_ref_i.reshape(-1)))
        if sim_idx == ref_idx:
            match += 1
        if not args.quiet_per_sample:
            print(f"[PLAN_SIM] sample {i}: sim_pred={sim_idx} ref_pred={ref_idx} max_abs_diff={diff}")

    total = end - start
    rate = (match / total) if total else 0.0
    print(f"[PLAN_SIM] done. samples={total} top1_match_rate={rate:.4f} max_abs_diff_max={max_abs}")

    if args.dump_output_npy is not None:
        import numpy as np

        out = np.stack(dumped_outputs, axis=0) if dumped_outputs else np.zeros((0,) + y_ref.shape[1:], dtype=np.float32)
        args.dump_output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(args.dump_output_npy), out.astype(np.float32))
        print(f"[PLAN_SIM] dumped outputs -> {args.dump_output_npy} shape={out.shape}")


if __name__ == "__main__":
    main()
