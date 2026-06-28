# -*- coding: utf-8 -*-
"""
Backend smoke test for compiling a hand-written 3x3 conv program end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

from venuscore_compiler import compile_program
from venuscore_compiler.frontend.manual_builder import build_single_conv3x3_program


def test_compile_conv3x3_end_to_end(tmp_path: Path) -> None:
    """Compile a tiny conv3x3 program and ensure artifacts are produced."""

    program = build_single_conv3x3_program(hin=8, win=8, cin=8, cout=8, padding=(1, 1, 1, 1), stride=(1, 1))
    artifact = compile_program(program, output_dir=tmp_path, dump_ir=True, dump_uop=True)

    uops_path = tmp_path / "uops.bin"
    params_path = tmp_path / "params.bin"
    metadata_path = tmp_path / "metadata.json"
    debug_ir = tmp_path / "debug_ir.json"
    debug_uops = tmp_path / "debug_uops.json"

    assert uops_path.exists() and uops_path.stat().st_size == len(artifact.uop_binary)
    assert params_path.exists() and params_path.stat().st_size == len(artifact.param_block)
    assert metadata_path.exists()
    meta = json.loads(metadata_path.read_text())
    if "tile_uop_map" in meta:
        tile_map = meta["tile_uop_map"]
    else:
        tile_map = meta.get("metadata", {}).get("tile_uop_map", [])
    assert tile_map != []
    # uop_binary size should be 32 bytes per uop
    assert len(artifact.uop_binary) % 32 == 0
    # debug dumps generated
    assert debug_ir.exists()
    assert debug_uops.exists()
