# -*- coding: utf-8 -*-
"""
Module overview:
  - Compiler configuration data structures.
  - Dependencies:
    * Depends on: dataclasses, pathlib.Path
    * Used by: cli to build configs for the compilation pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompileConfig:
    """User-facing configuration for a compilation run."""

    input_path: Path | None = None
    output_dir: Path = Path("out")
    target: str = "venuscore-v1"
    dump_ir: bool = False
    dump_uop: bool = False
