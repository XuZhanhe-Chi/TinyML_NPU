#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/xilinx.sh"
XILINX_SETTINGS="$(find_xilinx_settings Vitis)"
XSA="$REPO_ROOT/build/vivado_zybo7010/tinyml_npu_zybo7010.xsa"

if [[ ! -f "$XSA" ]]; then
  echo "Missing hardware platform: $XSA" >&2
  echo "Run scripts/build_zybo7010.sh first." >&2
  exit 2
fi

source "$XILINX_SETTINGS" >/dev/null 2>&1
xsct -nodisp "$REPO_ROOT/fpga/zybo7010/scripts/build_vitis.tcl" "$REPO_ROOT"

BUILD_DIR="$REPO_ROOT/build/vitis_zybo7010"
BSP_DIR="$BUILD_DIR/bsp/ps7_cortexa9_0"
TEMPLATE_DIR="$BUILD_DIR/app_template"
SOURCE_DIR="$REPO_ROOT/fpga/zybo7010/app/src"
OBJECT_DIR="$BUILD_DIR/obj"
ELF="$BUILD_DIR/kws_test.elf"
CC="${CC:-arm-none-eabi-gcc}"

mkdir -p "$OBJECT_DIR"
COMMON_FLAGS=(
  -mcpu=cortex-a9
  -mfpu=vfpv3
  -mfloat-abi=hard
  -O2
  -g
  -ffunction-sections
  -fdata-sections
  -Wall
  -Wextra
  -I"$BSP_DIR/include"
  -I"$SOURCE_DIR"
)

for source in main.c venus_driver.c; do
  "$CC" "${COMMON_FLAGS[@]}" -c "$SOURCE_DIR/$source" \
    -o "$OBJECT_DIR/${source%.c}.o"
done

"$CC" -o "$ELF" "$OBJECT_DIR/main.o" "$OBJECT_DIR/venus_driver.o" \
  -mcpu=cortex-a9 -mfpu=vfpv3 -mfloat-abi=hard \
  -Wl,-build-id=none -Wl,--gc-sections \
  -specs="$TEMPLATE_DIR/Xilinx.spec" \
  -T"$TEMPLATE_DIR/lscript.ld" \
  -L"$BSP_DIR/lib" \
  -Wl,--start-group -lxil -lm -lgcc -lc -Wl,--end-group

test -s "$ELF"
echo "TINYML_NPU_ELF=$ELF"
echo "TINYML_NPU_VITIS_PASS"
