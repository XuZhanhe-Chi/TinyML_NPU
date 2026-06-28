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
from venuscore_compiler.config import CompileConfig
from venuscore_compiler.ir.program import VcProgram


_DEF_TARGET = "venuscore-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="VenusCore NPU compiler")
    parser.add_argument("--input", type=str, required=True, help="Input model path")
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


def load_program(input_path: Path) -> VcProgram:
    """Load an ONNX model or build the hand-written smoke graph."""

    from venuscore_compiler.frontend import manual_builder
    from venuscore_compiler.frontend.onnx_loader import load_onnx_to_ir

    if input_path.suffix.lower() == ".onnx":
        return load_onnx_to_ir(str(input_path))

    return manual_builder.build_single_conv3x3_program(name=input_path.stem or "manual")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    logger = logging_utils.setup_logging()
    cfg = CompileConfig(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        target=args.target,
        dump_ir=args.dump_ir,
        dump_uop=args.dump_uop,
    )

    logger.info("Loading model from %s", cfg.input_path)
    program = load_program(cfg.input_path)
    logger.info("Compiling for target %s", cfg.target)
    compile_program(
        program,
        output_dir=cfg.output_dir,
        target=cfg.target,
        dump_ir=cfg.dump_ir,
        dump_uop=cfg.dump_uop,
    )
    logger.info("Compilation completed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
