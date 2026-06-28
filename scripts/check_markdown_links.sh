#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

failed=0
while IFS= read -r entry; do
  file="${entry%%$'\t'*}"
  target="${entry#*$'\t'}"
  target="${target%%#*}"
  [[ -z "$target" || "$target" == *://* || "$target" == mailto:* ]] && continue
  if [[ ! -e "$(dirname "$file")/$target" ]]; then
    echo "[DOCS] $file: missing link target $target" >&2
    failed=1
  fi
done < <(
  git ls-files '*.md' | while IFS= read -r file; do
    sed -nE 's/.*\[[^]]*\]\(([^)]+)\).*/\1/p' "$file" |
      while IFS= read -r target; do printf '%s\t%s\n' "$file" "$target"; done
  done
)

(( failed == 0 )) || exit 1
echo '[DOCS] Markdown link targets PASS'
