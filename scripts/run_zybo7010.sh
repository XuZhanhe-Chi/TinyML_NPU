#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XILINX_SETTINGS="${XILINX_SETTINGS:-/home/tools/Xilinx/Vitis/2021.1/settings64.sh}"

if [[ ! -f "$XILINX_SETTINGS" ]]; then
  echo "Missing Vitis settings file: $XILINX_SETTINGS" >&2
  exit 2
fi

source "$XILINX_SETTINGS" >/dev/null 2>&1
xsct -nodisp "$REPO_ROOT/fpga/zybo7010/scripts/run_board.tcl" "$REPO_ROOT"
