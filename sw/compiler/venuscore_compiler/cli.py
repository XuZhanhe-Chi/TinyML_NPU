# -*- coding: utf-8 -*-
"""
Module overview:
  - CLI argument parsing and entrypoint to run the VenusCore compiler pipeline.
  - Dependencies:
    * Depends on: venuscore_compiler.compile_program, logging_utils, frontend.*
    * Used by: console scripts / command-line invocation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from venuscore_compiler import compile_program
from venuscore_compiler import logging_utils
from venuscore_compiler.ir.program import VcProgram


_DEF_TARGET = "venuscore-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="VenusCore NPU compiler")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Input ONNX model path")
    source.add_argument(
        "--manual-smoke",
        action="store_true",
        help="Compile the built-in single Conv3x3 smoke graph",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="Directory for compiled artifacts",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=_DEF_TARGET,
        help="Target VenusCore NPU variant (e.g. venuscore-v1)",
    )
    parser.add_argument(
        "--dump-ir",
        action="store_true",
        help="Dump intermediate representation to JSON",
    )
    parser.add_argument(
        "--dump-uop",
        action="store_true",
        help="Dump encoded uOPs for debugging",
    )
    return parser.parse_args(argv)


def load_program(input_path: Path | None, *, manual_smoke: bool = False) -> VcProgram:
    """Load an existing ONNX model or explicitly build the smoke graph."""

    from venuscore_compiler.frontend import manual_builder
    from venuscore_compiler.frontend.onnx_loader import load_onnx_to_ir

    if manual_smoke:
        return manual_builder.build_single_conv3x3_program(name="manual_smoke")
    if input_path is None:
        raise ValueError("An ONNX input path or --manual-smoke is required.")
    if input_path.suffix.lower() != ".onnx":
        raise ValueError(f"Input must be an .onnx model, got: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input model does not exist: {input_path}")

    return load_onnx_to_ir(str(input_path))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    logger = logging_utils.setup_logging()
    input_path = args.input
    logger.info("Loading %s", input_path if input_path is not None else "manual smoke graph")
    try:
        program = load_program(input_path, manual_smoke=args.manual_smoke)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    output_dir = Path(args.output_dir)
    logger.info("Compiling for target %s", args.target)
    compile_program(
        program,
        output_dir=output_dir,
        target=args.target,
        dump_ir=args.dump_ir,
        dump_uop=args.dump_uop,
    )
    logger.info("Compilation completed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
