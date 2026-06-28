#!/usr/bin/env bash
set -euo pipefail

TOP="VenusCoreTop"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --top)
      TOP="${2:?missing value for --top}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--top VenusCoreTop|VenusCoreTopBB]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$TOP" in
  VenusCoreTop|VenusCoreTopBB)
    ;;
  *)
    echo "Unsupported top: $TOP" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/build/rtl"

cd "$REPO_ROOT/hw/spinal"
sbt -batch "runMain venuscore.top.${TOP}"

echo "[INFO] RTL generated under $REPO_ROOT/build/rtl"
