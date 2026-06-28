#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo '[INFO] scripts/check_local.sh is a compatibility wrapper for make check.'
exec make -C "$REPO_ROOT" check "$@"
