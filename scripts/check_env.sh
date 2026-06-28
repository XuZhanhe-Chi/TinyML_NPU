#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/xilinx.sh"

CHECK_BOARD=0
STRICT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --board) CHECK_BOARD=1 ;;
    --strict) STRICT=1 ;;
    -h|--help)
      echo "Usage: $0 [--board] [--strict]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

missing=0
ok() { printf '[OK]      %-14s %s\n' "$1" "$2"; }
warn() { printf '[MISSING] %-14s %s\n' "$1" "$2"; missing=$((missing + 1)); }

check_cmd() {
  local command_name="$1"
  local display_name="${2:-$1}"
  if command -v "$command_name" >/dev/null 2>&1; then
    ok "$display_name" "$(command -v "$command_name")"
  else
    warn "$display_name" "command not found"
  fi
}

check_cmd python3 Python
check_cmd java Java
check_cmd sbt sbt
check_cmd iverilog Icarus
check_cmd gcc GCC
check_cmd git Git

if command -v python3 >/dev/null 2>&1; then
  if python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    ok 'Python version' "$(python3 --version 2>&1)"
  else
    warn 'Python version' "Python 3.10 or newer is required"
  fi
fi

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  ok 'Project venv' "$REPO_ROOT/.venv"
else
  printf '[INFO]    %-14s %s\n' 'Project venv' 'not created; run make setup'
fi

if (( CHECK_BOARD )); then
  for tool in Vivado Vitis; do
    if settings="$(find_xilinx_settings "$tool" 2>/dev/null)"; then
      ok "$tool settings" "$settings"
      expected_cmd="${tool,,}"
      [[ "$tool" == "Vitis" ]] && expected_cmd="xsct"
      if bash -c 'source "$1" >/dev/null 2>&1 && command -v "$2"' _ "$settings" "$expected_cmd" >/dev/null; then
        ok "$expected_cmd" "available after sourcing settings"
      else
        warn "$expected_cmd" "not available after sourcing $settings"
      fi
    else
      warn "$tool settings" "set XILINX_${tool^^}_SETTINGS"
    fi
  done

  if command -v lsusb >/dev/null 2>&1; then
    if lsusb | grep -Eqi 'Digilent|Xilinx|FTDI|Future Technology Devices'; then
      ok 'JTAG USB' "candidate programming cable detected"
    else
      printf '[INFO]    %-14s %s\n' 'JTAG USB' 'no known cable string detected'
    fi
  else
    printf '[INFO]    %-14s %s\n' 'JTAG USB' 'lsusb unavailable; use make zybo-probe'
  fi
fi

if (( missing > 0 )); then
  echo "[ENV] $missing required item(s) missing."
  (( STRICT )) && exit 1
else
  echo '[ENV] Required tools detected.'
fi
