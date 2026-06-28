#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/sim"
mkdir -p "$BUILD_DIR"

for name in axi_lite_to_apb3 ahb_lite_to_bram_port; do
  echo "[RTL] Running $name protocol regression"
  iverilog -g2012 -Wall -Wno-timescale \
    -s "tb_${name}" \
    -o "$BUILD_DIR/${name}.vvp" \
    "$REPO_ROOT/fpga/zybo7010/rtl/${name}.v" \
    "$REPO_ROOT/fpga/zybo7010/sim/tb_${name}.v"
  vvp "$BUILD_DIR/${name}.vvp"
done

echo '[RTL] Glue regressions PASS'
