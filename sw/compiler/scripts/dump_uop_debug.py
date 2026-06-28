# -*- coding: utf-8 -*-
"""
Module overview:
  - Debug script entrypoint for decoding uops.bin.
  - Dependencies:
    * Depends on: venuscore_compiler.isa.decoder
    * Used by: developers when debugging uOP streams
"""

from __future__ import annotations

import argparse
from pathlib import Path

from venuscore_compiler.isa.decoder import decode_uops


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode uops.bin and print semantic uOPs.")
    p.add_argument("--uops-bin", type=Path, default=Path("uops.bin"), help="Path to uops.bin (default: ./uops.bin)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    uop_path: Path = args.uops_bin
    if not uop_path.exists():
        raise SystemExit(f"uops.bin not found: {uop_path}")
    data = uop_path.read_bytes()
    # Decode sequentially and print an index to correlate with encoded order.
    uops = decode_uops(data)
    for idx, uop in enumerate(uops):
        print(idx, uop)


if __name__ == "__main__":
    main()


if __name__ == "__main__":  # pragma: no cover
    main()
