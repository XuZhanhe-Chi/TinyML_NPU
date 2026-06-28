# -*- coding: utf-8 -*-
"""
Module overview:
  - Dumps IR and uOP data to JSON for debugging.
  - Dependencies:
    * Depends on: venuscore_compiler.ir.program, venuscore_compiler.isa.uop_format
    * Used by: compile_program and debug scripts
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from venuscore_compiler.ir.program import VcProgram
from venuscore_compiler.isa.uop_format import Uop


def dump_program_to_json(program: VcProgram, path: str | Path) -> None:
    """Serialize the program to JSON for inspection."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Store a human-readable snapshot of the IR to help trace compilation steps.
    out_path.write_text(json.dumps(program.to_dict(), indent=2))


def dump_ir(program: VcProgram, path: str | Path) -> None:
    """Write IR to a JSON file using the program's to_dict() representation."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(program.to_dict(), indent=2))


def dump_uops(uops: Iterable[Uop], path: str | Path) -> None:
    """Serialize uOPs to JSON for inspection."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist logical uOP view; physical encoding lives in isa.encoder.
    out_path.write_text(json.dumps([uop.to_dict() for uop in uops], indent=2))
