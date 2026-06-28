#!/usr/bin/env bash

find_xilinx_settings() {
  local tool="${1:?tool name required}"
  local specific_var=""
  local candidate=""

  case "$tool" in
    Vivado) specific_var="XILINX_VIVADO_SETTINGS" ;;
    Vitis) specific_var="XILINX_VITIS_SETTINGS" ;;
    *) echo "Unsupported Xilinx tool: $tool" >&2; return 2 ;;
  esac

  candidate="${!specific_var:-}"
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  if [[ -n "${XILINX_SETTINGS:-}" && -f "$XILINX_SETTINGS" ]]; then
    printf '%s\n' "$XILINX_SETTINGS"
    return 0
  fi

  for candidate in \
    "/home/tools/Xilinx/$tool/2021.1/settings64.sh" \
    "/opt/Xilinx/$tool/2021.1/settings64.sh"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "Unable to find $tool 2021.1 settings64.sh." >&2
  echo "Set $specific_var to the full settings64.sh path." >&2
  return 2
}
