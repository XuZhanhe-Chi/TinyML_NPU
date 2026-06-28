# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from venuscore_compiler import cli


def test_cli_requires_explicit_input_mode() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.parse_args([])
    assert exc.value.code == 2


def test_cli_manual_smoke_compiles(tmp_path: Path) -> None:
    rc = cli.main(["--manual-smoke", "--output-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "uops.bin").is_file()
    assert (tmp_path / "params.bin").is_file()
    assert (tmp_path / "bundle.h").is_file()


def test_cli_rejects_missing_or_non_onnx_input(tmp_path: Path) -> None:
    assert cli.main(["--input", str(tmp_path / "missing.onnx")]) == 2
    bad = tmp_path / "model.txt"
    bad.write_text("not a model", encoding="utf-8")
    assert cli.main(["--input", str(bad)]) == 2
