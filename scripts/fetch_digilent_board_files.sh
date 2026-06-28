#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="36f34ab687b7fa9c778b779d027f3bce63b3ace9"
DEST="${DIGILENT_BOARD_CHECKOUT:-$REPO_ROOT/build/deps/vivado-boards}"

validate_board_dir() {
  local board_dir="$1"
  [[ -f "$board_dir/zybo/B.4/board.xml" || -f "$board_dir/zybo/board.xml" ]]
}

if [[ -n "${DIGILENT_BOARD_REPO:-}" ]]; then
  if ! validate_board_dir "$DIGILENT_BOARD_REPO"; then
    echo "DIGILENT_BOARD_REPO is not a valid board_files directory: $DIGILENT_BOARD_REPO" >&2
    exit 2
  fi
  printf '%s\n' "$(cd "$DIGILENT_BOARD_REPO" && pwd)"
  exit 0
fi

if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  echo "[DEPS] Cloning Digilent/vivado-boards into $DEST" >&2
  git clone --filter=blob:none --no-checkout \
    https://github.com/Digilent/vivado-boards.git "$DEST"
fi

if ! git -C "$DEST" cat-file -e "$PIN^{commit}" 2>/dev/null; then
  git -C "$DEST" fetch --depth 1 origin "$PIN" >&2
fi
git -C "$DEST" checkout --detach --force "$PIN" >&2

BOARD_DIR="$DEST/new/board_files"
if ! validate_board_dir "$BOARD_DIR"; then
  echo "Pinned Digilent checkout does not contain the original Zybo board definition." >&2
  exit 2
fi

printf '%s\n' "$BOARD_DIR"
