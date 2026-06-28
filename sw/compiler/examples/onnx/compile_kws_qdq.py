# -*- coding: utf-8 -*-
"""
Example: compile the QDQ INT8 KWS ONNX model end-to-end.

This script loads a user-provided KWS ONNX model with the ONNX frontend and runs
the full VenusCore compiler pipeline, producing uops/params/metadata under
``out/examples/onnx_kws_qdq`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from venuscore_compiler import compile_program
from venuscore_compiler.config import default_hw_config
from venuscore_compiler.frontend.onnx_loader import load_onnx_to_ir


def _apply_pool_round_overrides(program, op_names: list[str]) -> list[str]:
    if not op_names:
        return []

    touched: list[str] = []
    targets = set(op_names)
    for op in getattr(program, "ops", []):
        if getattr(op, "name", None) not in targets:
            continue
        if getattr(op, "op_type", None) not in ("avg_pool", "avgpool"):
            raise ValueError(
                f"--pool-ties-even-op expects an avg_pool op, got name='{op.name}' op_type='{op.op_type}'"
            )
        op.qmode = "INT4"
        touched.append(op.name)

    missing = [name for name in op_names if name not in touched]
    if missing:
        raise ValueError(f"Failed to find requested pool ops for ties-even override: {missing}")
    return touched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile QDQ INT8 KWS ONNX model for VenusCore.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to QDQ INT8 KWS ONNX model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/examples/onnx_kws_qdq"),
        help="Directory to store compiled artifacts (default: out/examples/onnx_kws_qdq)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="venuscore-v1",
        help="Target name for compilation (default: venuscore-v1)",
    )
    parser.add_argument(
        "--hw-target",
        type=str,
        default="zybo7010",
        help="Hardware target preset for tiling/capacity (default: zybo7010)",
    )
    parser.add_argument(
        "--address-mode",
        type=str,
        choices=["offset", "absolute"],
        default="offset",
        help="Addressing mode for uOPs/plan (default: offset)",
    )
    parser.add_argument(
        "--dump-debug",
        action="store_true",
        help="If set, also dump IR/uOP debug JSON files.",
    )
    parser.add_argument(
        "--no-post-check",
        action="store_true",
        help="Disable host-side post-compile checks (bundle/relocation/plan sanity).",
    )
    parser.add_argument(
        "--post-check-act-base",
        type=lambda s: int(s, 0),
        default=0x2000_0000,
        help="Activation base used by post-check in offset mode (default: 0x20000000).",
    )
    parser.add_argument(
        "--post-check-param-base",
        type=lambda s: int(s, 0),
        default=0x2100_0000,
        help="Param base used by post-check in offset mode (default: 0x21000000).",
    )
    parser.add_argument(
        "--pool-ties-even-op",
        type=str,
        action="append",
        default=[],
        help="AvgPool op name to compile with ties-to-even divide-by-4 semantics. Can be specified multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path: Path = args.model
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {model_path}")

    program = load_onnx_to_ir(str(model_path))
    pool_ties_even_ops = _apply_pool_round_overrides(program, args.pool_ties_even_op)
    if args.address_mode == "offset":
        ifm_base = ofm_base = param_base = None
    else:
        ifm_base = 0x0000_0000
        ofm_base = 0x0008_0000
        param_base = 0x0010_0000

    artifact = compile_program(
        program,
        output_dir=output_dir,
        target=args.target,
        hw=default_hw_config(args.hw_target),
        dump_ir=args.dump_debug,
        dump_uop=args.dump_debug,
        ifm_base=ifm_base,
        ofm_base=ofm_base,
        param_base=param_base,
        post_check=not args.no_post_check,
        post_check_act_base=args.post_check_act_base,
        post_check_param_base=args.post_check_param_base,
    )

    metadata_path = output_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    print("Compile finished.")
    print(f"  model       : {model_path}")
    print(f"  output_dir  : {output_dir}")
    print(f"  uops.bin    : {len(artifact.uop_binary)} bytes")
    print(f"  params.bin  : {len(artifact.param_block)} bytes")
    print(f"  metadata.json: {metadata_path.exists()}")
    mapped = getattr(program, "metadata", {}).get("onnx_mapped_nodes", [])
    skipped = getattr(program, "metadata", {}).get("onnx_skipped_nodes", [])
    print(f"  ONNX mapped nodes  : {len(mapped)}")
    print(f"  ONNX skipped nodes : {len(skipped)}")
    if pool_ties_even_ops:
        print(f"  pool ties-even ops : {', '.join(pool_ties_even_ops)}")
    if skipped:
        print("  Skipped ops:", ", ".join(skipped))
    if metadata:
        meta_core = metadata.get("metadata", metadata)
        peak_act = meta_core.get("activation_peak_bytes")
        weight_bytes = meta_core.get("weight_bytes")
        output_base = meta_core.get("output_base")
        param_size = metadata.get("param_block_size")
        uop_size = metadata.get("uop_binary_size")
        print(f"  activation_peak_bytes : {peak_act}")
        print(f"  weight_bytes          : {weight_bytes}")
        print(f"  output_base           : {output_base}")
        print(f"  param_block_size      : {param_size}")
        print(f"  uop_binary_size       : {uop_size}")


if __name__ == "__main__":
    main()
