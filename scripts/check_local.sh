#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[CHECK] Python package and tests"
(
  cd sw/compiler
  python -m pip install -e .[dev]
  pytest -q
  python -m examples.hand_written.conv3x3_single_tile --output-dir out/examples/conv3x3_single_tile
  python -m scripts.check_bundle_relocation \
    --out-dir out/examples/conv3x3_single_tile \
    --act-base 0x40000000 \
    --param-base 0x40000000
)

echo "[CHECK] SpinalHDL compile"
(
  cd hw/spinal
  sbt -batch compile
)

echo "[CHECK] RTL generation"
bash scripts/gen_rtl.sh --top VenusCoreTop
bash scripts/gen_rtl.sh --top VenusCoreTopBB

if command -v iverilog >/dev/null 2>&1; then
  echo "[CHECK] Verilog syntax"
  iverilog -g2005-sv -tnull -o /tmp/tinyml_npu_zybo_check.vvp \
    fpga/zybo7010/rtl/*.v \
    build/rtl/VenusCoreTop.v \
    build/rtl/VenusCoreTopBB.v
else
  echo "[WARN] iverilog not found; skipping Verilog syntax check"
fi

echo "[CHECK] Firmware C syntax"
TMPDIR_STUB="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_STUB"' EXIT
cat > "$TMPDIR_STUB/xparameters.h" <<'EOF'
#ifndef XPARAMETERS_H
#define XPARAMETERS_H
#define XPAR_VENUSCORETOP_0_BASEADDR 0x43C00000u
#endif
EOF
cat > "$TMPDIR_STUB/xil_cache.h" <<'EOF'
#ifndef XIL_CACHE_H
#define XIL_CACHE_H
#include <stdint.h>
typedef uintptr_t INTPTR;
static inline void Xil_DCacheFlushRange(INTPTR addr, unsigned len) {(void)addr; (void)len;}
static inline void Xil_DCacheInvalidateRange(INTPTR addr, unsigned len) {(void)addr; (void)len;}
#endif
EOF
gcc -std=c11 -Wall -Wextra \
  -I "$TMPDIR_STUB" \
  -I fpga/zybo7010/app/src \
  -fsyntax-only \
  fpga/zybo7010/app/src/main.c \
  fpga/zybo7010/app/src/venus_driver.c

echo "[CHECK] Public tree hygiene"
bash scripts/check_public_tree.sh

echo "[CHECK] PASS"
