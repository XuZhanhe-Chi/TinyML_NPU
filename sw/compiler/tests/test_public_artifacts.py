# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from venuscore_compiler.common import capacity
from venuscore_compiler.config import default_hw_config
from venuscore_compiler.runtime.bundle_h_parser import load_bundle_h


REPO_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = REPO_ROOT / "fpga" / "zybo7010" / "app"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _macro(text: str, name: str) -> int:
    match = re.search(rf"^#define\s+{name}\s+(0x[0-9A-Fa-f]+|\d+)[uU]?", text, re.MULTILINE)
    assert match, f"macro not found: {name}"
    return int(match.group(1), 0)


def _tcl_value(text: str, name: str) -> int:
    match = re.search(rf"^set\s+{name}\s+(0x[0-9A-Fa-f]+|\d+)$", text, re.MULTILINE)
    assert match, f"Tcl constant not found: {name}"
    return int(match.group(1), 0)


def test_committed_kws_bundle_contract() -> None:
    bundle = load_bundle_h(APP_DIR / "src" / "bundle.h")
    assert bundle.define_int("ADDRESS_MODE_OFFSET") == 1
    assert bundle.define_int("UOPS_LEN_BYTES") == 1408
    assert bundle.define_int("PARAMS_LEN_BYTES") == 58112
    assert bundle.define_int("ACTIVATION_PEAK_BYTES") == 64000
    assert bundle.define_int("INPUT_SIZE") == 8000
    assert bundle.define_int("OUTPUT_BASE") == 32000
    assert bundle.define_int("OUTPUT_SIZE") == 12
    assert len(bundle.uops_words) == 44 * 8
    assert len(bundle.params_words) * 4 == 58112


def test_demo_artifact_manifest() -> None:
    manifest = json.loads((APP_DIR / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["model_distributed"] is False
    for artifact in manifest["artifacts"]:
        path = APP_DIR / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]


def test_zybo_target_capacity_matches_public_contract() -> None:
    hw = default_hw_config("zybo7010")
    assert hw.cluster_count == 1
    assert hw.ibuf_line_bytes == capacity.IBUF_LINE_BYTES == 3840
    assert capacity.IBUF_TOTAL_BYTES == 12 * 1024
    assert hw.wbuf_lane_bytes == capacity.WBUF_LANE_BYTES == 2048
    assert hw.wbuf_lanes == capacity.WBUF_LANES == 4
    assert hw.wbuf_lane_bytes * hw.wbuf_lanes == 8 * 1024

    scala = (REPO_ROOT / "hw/spinal/src/main/scala/venuscore/config/VenusCoreConfig.scala").read_text()
    assert "sharedMemSizeBytes: Int = 128 * 1024" in scala
    assert "clusterNum: Int = 1" in scala
    assert "lanePerCluster: Int = 4" in scala
    assert "simdPerLane: Int = 4" in scala
    assert "wbufSizeBytes: Int = 2048" in scala
    assert "ibufSizeBytes: Int = 3072 * 4" in scala


def test_board_result_abi_matches_xsdb_and_docs() -> None:
    header = (APP_DIR / "src" / "board_result.h").read_text()
    tcl = (REPO_ROOT / "fpga/zybo7010/scripts/run_board.tcl").read_text()
    docs = (REPO_ROOT / "docs/isa-and-runtime.md").read_text()

    assert _macro(header, "TINYML_NPU_RESULT_ADDR") == _tcl_value(tcl, "result_addr")
    assert _macro(header, "TINYML_NPU_RESULT_MAGIC") == _tcl_value(tcl, "result_magic")
    assert _macro(header, "TINYML_NPU_EXPECTED_VERSION") == _tcl_value(tcl, "expected_version")
    assert _macro(header, "TINYML_NPU_EXPECTED_SAMPLE_COUNT") == _tcl_value(tcl, "expected_sample_count")
    assert _macro(header, "TINYML_NPU_EXPECTED_REF_TOP1_MATCH") == _tcl_value(tcl, "expected_ref_top1_match")
    assert _macro(header, "TINYML_NPU_MIN_LABEL_CORRECT") == _tcl_value(tcl, "min_label_correct")
    assert _macro(header, "TINYML_NPU_MAX_ABS_ERROR") == _tcl_value(tcl, "max_allowed_error")
    assert _macro(header, "TINYML_NPU_RESULT_BYTES") == 64
    assert _macro(header, "TINYML_NPU_RESULT_ABI_VERSION") == 2

    fields = re.search(
        r"typedef struct \{(?P<body>.*?)\} tinyml_npu_board_result_t;",
        header,
        re.DOTALL,
    )
    assert fields
    names = re.findall(r"uint32_t\s+(\w+)(?:\[\d+\])?;", fields.group("body"))
    assert names == [
        "magic", "code", "hw_status", "sample_count", "label_correct",
        "ref_top1_match", "max_abs_error", "total_mismatches",
        "first_failure_sample", "first_failure_top1",
        "first_failure_expected_top1", "first_failure_label",
        "status", "debug0", "debug1", "total_cycles",
    ]
    for name in names:
        assert f"| {name} |" in docs


def _u8_array_values(text: str, name: str) -> list[int]:
    match = re.search(
        rf"static\s+const\s+uint8_t\s+{name}\[[^\]]+\]\s*=\s*\{{(?P<body>.*?)\}};",
        text,
        re.DOTALL,
    )
    assert match, f"array not found: {name}"
    return [int(x, 16) for x in re.findall(r"0x([0-9A-Fa-f]{2})u", match.group("body"))]


def test_kws_multivector_header_is_balanced_and_large_enough() -> None:
    header = (APP_DIR / "src" / "kws_testvector_fpga.h").read_text()

    assert _macro(header, "VC_KWS_SAMPLE_COUNT") == 120
    assert _macro(header, "VC_KWS_MIN_SAMPLE_COUNT") == 100
    assert _macro(header, "VC_KWS_CLASS_COUNT") == 12
    assert _macro(header, "VC_KWS_EXPECTED_LABEL_CORRECT") == 117
    assert _macro(header, "VC_KWS_EXPECTED_REF_TOP1_MATCH") == 120

    labels = _u8_array_values(header, "VC_KWS_LABELS")
    expected_top1 = _u8_array_values(header, "VC_KWS_EXPECTED_TOP1")
    assert len(labels) == 120
    assert len(expected_top1) == 120
    assert sorted(set(labels)) == list(range(12))
    assert {label: labels.count(label) for label in range(12)} == {label: 10 for label in range(12)}
    assert sum(int(a == b) for a, b in zip(labels, expected_top1)) == 117

    input_samples = len(re.findall(r"/\* sample \d{3} \*/", header))
    assert input_samples == 240  # input samples + expected-output samples
