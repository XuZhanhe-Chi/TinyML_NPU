#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/xilinx.sh"
XILINX_SETTINGS="$(find_xilinx_settings Vitis)"

source "$XILINX_SETTINGS" >/dev/null 2>&1
xsct -nodisp "$REPO_ROOT/fpga/zybo7010/scripts/run_board.tcl" "$REPO_ROOT"
