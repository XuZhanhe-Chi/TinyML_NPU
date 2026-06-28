#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/xilinx.sh"
SETTINGS="${1:-$(find_xilinx_settings Vivado)}"

if [[ ! -f "$SETTINGS" ]]; then
  echo "Missing Xilinx settings file: $SETTINGS" >&2
  exit 2
fi

TMP_TCL="$(mktemp)"
trap 'rm -f "$TMP_TCL"' EXIT

cat > "$TMP_TCL" <<'EOF'
open_hw_manager
connect_hw_server -url localhost:3121
set targets [get_hw_targets]
puts "HW_TARGETS_BEGIN"
foreach target $targets { puts $target }
puts "HW_TARGETS_END"
foreach target $targets {
  current_hw_target $target
  if {![catch {open_hw_target}]} {
    puts "HW_DEVICES_BEGIN"
    foreach device [get_hw_devices] { puts $device }
    puts "HW_DEVICES_END"
    close_hw_target
  }
}
disconnect_hw_server
close_hw_manager
exit
EOF

source "$SETTINGS" >/dev/null 2>&1
vivado -mode batch -nojournal -nolog -source "$TMP_TCL"
