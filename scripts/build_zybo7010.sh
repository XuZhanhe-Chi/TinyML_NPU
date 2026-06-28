#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/xilinx.sh"
XILINX_SETTINGS="$(find_xilinx_settings Vivado)"
BOARD_REPO="${1:-}"

if [[ -z "$BOARD_REPO" ]]; then
  BOARD_REPO="$(bash "$REPO_ROOT/scripts/fetch_digilent_board_files.sh")"
elif [[ ! -d "$BOARD_REPO" ]]; then
  echo "Digilent board_files directory does not exist: $BOARD_REPO" >&2
  exit 2
fi

cd "$REPO_ROOT"
bash scripts/gen_rtl.sh --top VenusCoreTop
rm -rf "$REPO_ROOT/build/vivado_zybo7010"

source "$XILINX_SETTINGS" >/dev/null 2>&1
vivado -mode batch -nojournal -nolog \
  -source fpga/zybo7010/scripts/create_project.tcl \
  -tclargs "$REPO_ROOT" "$BOARD_REPO"
