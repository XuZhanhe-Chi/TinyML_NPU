#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

gcc -std=c11 -Wall -Wextra -Werror \
  -I "$TMPDIR_STUB" \
  -I "$REPO_ROOT/fpga/zybo7010/app/src" \
  -fsyntax-only \
  "$REPO_ROOT/fpga/zybo7010/app/src/main.c" \
  "$REPO_ROOT/fpga/zybo7010/app/src/venus_driver.c"

echo '[CHECK] Firmware syntax PASS'
