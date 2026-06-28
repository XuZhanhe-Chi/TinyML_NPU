#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-v0.1.0}"

if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Release version must look like v0.1.0, got: $VERSION" >&2
  exit 2
fi

cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Release packaging requires a clean Git worktree." >&2
  git status --short >&2
  exit 2
fi

BIT="build/vivado_zybo7010/tinyml_npu_zybo7010.bit"
XSA="build/vivado_zybo7010/tinyml_npu_zybo7010.xsa"
ELF="build/vitis_zybo7010/kws_test.elf"
TIMING="build/vivado_zybo7010/timing_summary.rpt"
UTIL="build/vivado_zybo7010/utilization.rpt"
CHECK_LOG="build/logs/check.log"
VIVADO_LOG="build/logs/vivado-final.log"
VITIS_LOG="build/logs/vitis.log"
BOARD_LOG="build/logs/board.log"

for required in "$BIT" "$XSA" "$ELF" "$TIMING" "$UTIL" \
  "$CHECK_LOG" "$VIVADO_LOG" "$VITIS_LOG" "$BOARD_LOG"; do
  if [[ ! -s "$required" ]]; then
    echo "Required release evidence is missing: $required" >&2
    exit 2
  fi
done

grep -Fq '[CHECK] ALL PASS' "$CHECK_LOG"
grep -Fq 'TINYML_NPU_VIVADO_PASS' "$VIVADO_LOG"
grep -Fq 'TINYML_NPU_UNROUTED_NETS=0' "$VIVADO_LOG"
grep -Fq 'TINYML_NPU_PARTIAL_NETS=0' "$VIVADO_LOG"
grep -Fq 'TINYML_NPU_VITIS_PASS' "$VITIS_LOG"
grep -Fq 'TINYML_NPU_VERSION=0x00050000' "$BOARD_LOG"
grep -Fq 'TINYML_NPU_RESULT code=0' "$BOARD_LOG"
grep -Fq 'top1=0' "$BOARD_LOG"
grep -Fq 'max_abs_error=5' "$BOARD_LOG"
grep -Fq 'TINYML_NPU_BOARD_PASS' "$BOARD_LOG"
grep -Fq 'checking no_clock (0)' "$TIMING"
grep -Fq 'checking unconstrained_internal_endpoints (0)' "$TIMING"

WNS="$(sed -n 's/.*TINYML_NPU_WNS=//p' "$VIVADO_LOG" | tail -1)"
if [[ -z "$WNS" ]] || ! awk -v value="$WNS" 'BEGIN { exit !(value >= 0.0) }'; then
  echo "Invalid or negative WNS in Vivado evidence: $WNS" >&2
  exit 2
fi

OUT="build/release/$VERSION"
rm -rf "$OUT"
mkdir -p "$OUT"

cp "$BIT" "$OUT/tinyml_npu_zybo7010_${VERSION}.bit"
cp "$XSA" "$OUT/tinyml_npu_zybo7010_${VERSION}.xsa"
cp "$ELF" "$OUT/tinyml_npu_kws_test_${VERSION}.elf"
RELEASE_NOTES="docs/releases/${VERSION}.md"
if [[ ! -f "$RELEASE_NOTES" ]]; then
  echo "Release notes are missing: $RELEASE_NOTES" >&2
  exit 2
fi
cp "$RELEASE_NOTES" "$OUT/RELEASE_NOTES.md"
cp "THIRD_PARTY_NOTICES.md" "$OUT/THIRD_PARTY_NOTICES.md"

cat > "$OUT/VERIFICATION.txt" <<EOF
TinyML_NPU $VERSION verification summary
git commit: $(git rev-parse HEAD)
board: original Digilent Zybo, xc7z010clg400-1
PL clock: 50 MHz
Vivado/Vitis: 2021.1
WNS: $WNS ns
unrouted nets: 0
partially routed nets: 0
no_clock checks: 0
unconstrained internal endpoints: 0
NPU version: 0x00050000
KWS uOPs: 44
top1: 0 (expected 0)
byte mismatches: 10/12
maximum INT8 absolute error: 5 (tolerance 5)
NPU active cycles: 563188
active-cycle equivalent: 11.26376 ms at 50 MHz
board result: PASS

The active-cycle value excludes PS setup, cache operations, UART, and JTAG.
EOF

export RELEASE_OUT="$OUT"
export RELEASE_VERSION="$VERSION"
export RELEASE_COMMIT="$(git rev-parse HEAD)"
export RELEASE_WNS="$WNS"
python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

out = Path(os.environ["RELEASE_OUT"])
asset_kinds = {
    ".bit": "Vivado-generated FPGA configuration",
    ".xsa": "Vivado-generated hardware platform",
    ".elf": "Vitis-generated bare-metal application",
    ".md": "documentation",
    ".txt": "verification record",
}
assets = []
for path in sorted(out.iterdir()):
    if path.name in {"MANIFEST.json", "SHA256SUMS"}:
        continue
    data = path.read_bytes()
    assets.append(
        {
            "file": path.name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "kind": asset_kinds.get(path.suffix, "release asset"),
        }
    )

manifest = {
    "schema_version": 1,
    "project": "TinyML_NPU",
    "accelerator": "VenusCore",
    "version": os.environ["RELEASE_VERSION"],
    "git_commit": os.environ["RELEASE_COMMIT"],
    "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "source_license": "Apache-2.0",
    "generated_artifact_notice": "See THIRD_PARTY_NOTICES.md; vendor terms may apply.",
    "target": {
        "board": "original Digilent Zybo",
        "device": "xc7z010clg400-1",
        "pl_clock_mhz": 50,
        "vivado_vitis": "2021.1",
    },
    "verification": {
        "wns_ns": float(os.environ["RELEASE_WNS"]),
        "unrouted_nets": 0,
        "unconstrained_internal_endpoints": 0,
        "npu_version": "0x00050000",
        "top1": 0,
        "max_abs_error": 5,
        "npu_active_cycles": 563188,
        "board_pass": True,
    },
    "assets": assets,
}
(out / "MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
)
PY

(
  cd "$OUT"
  sha256sum ./* | sort -k2 > SHA256SUMS
  sha256sum -c SHA256SUMS
)

echo "TINYML_NPU_RELEASE_DIR=$REPO_ROOT/$OUT"
echo "TINYML_NPU_RELEASE_COMMIT=$(git rev-parse HEAD)"
echo "TINYML_NPU_RELEASE_PACKAGE_PASS"
