#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XILINX_SETTINGS="${XILINX_SETTINGS:-/home/tools/Xilinx/Vivado/2021.1/settings64.sh}"
BOARD_REPO="${DIGILENT_BOARD_REPO:-${1:-}}"

if [[ ! -f "$XILINX_SETTINGS" ]]; then
  echo "Missing Vivado settings file: $XILINX_SETTINGS" >&2
  exit 2
fi
if [[ -z "$BOARD_REPO" || ! -d "$BOARD_REPO" ]]; then
  echo "Set DIGILENT_BOARD_REPO or pass the Digilent board_files directory." >&2
  exit 2
fi

cd "$REPO_ROOT"
bash scripts/gen_rtl.sh --top VenusCoreTop
rm -rf "$REPO_ROOT/build/vivado_zybo7010"

source "$XILINX_SETTINGS" >/dev/null 2>&1
vivado -mode batch -nojournal -nolog \
  -source fpga/zybo7010/scripts/create_project.tcl \
  -tclargs "$REPO_ROOT" "$BOARD_REPO"
