#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[CHECK] Files larger than 10MB"
large_files=()
while IFS= read -r -d '' file; do
  if (( $(stat -c '%s' "$file") > 10 * 1024 * 1024 )); then
    large_files+=("$file")
  fi
done < <(git ls-files --cached --others --exclude-standard -z)
if (( ${#large_files[@]} > 0 )); then
  printf '%s\n' "${large_files[@]}" >&2
  echo "[CHECK] Public tree contains files larger than 10MB" >&2
  exit 1
fi

echo "[CHECK] Scope keywords"
PATTERN="$(printf '%s|%s|%s|%s|%s|%s|%s|%s' "AS""IC" "40""nm" "PT""PX" "SP""EF" "FS""DB" "Go""win" "GW""5A" "Vex""Riscv")"
if rg -n "$PATTERN" . \
  --glob '!.git/**' --glob '!build/**'; then
  echo "[CHECK] Scope keyword scan found matches" >&2
  exit 1
fi

echo "[CHECK] PASS"
