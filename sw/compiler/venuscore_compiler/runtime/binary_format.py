# -*- coding: utf-8 -*-
"""
Binary packaging for VenusCore compiled artifacts.

Contains:
  - uop_binary: contiguous 32-byte little-endian uOP stream.
  - param_block: contiguous Param Block blob (Quant Coeff + aligned weight data) produced by layout_param_block.
  - metadata: auxiliary info (target, counts, sizes, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from venuscore_compiler.isa.uop_format import Uop


@dataclass
class CompiledArtifact:
    """
    Container for compiler outputs.

    Fields:
      - uops: semantic uOP list (logical view).
      - uop_binary: encoded uOP stream (32B each, packed by encoder).
      - param_block: Param Block blob (concatenated per-tile blocks).
      - metadata: optional info (target, sizes).
    """

    uops: List[Uop] = field(default_factory=list)
    uop_binary: bytes = b""
    param_block: bytes = b""
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Convert to a serializable dictionary (logical view + sizes)."""

        return {
            "metadata": self.metadata,
            "uops": [uop.to_dict() for uop in self.uops],
            "param_block_size": len(self.param_block),
            "uop_binary_size": len(self.uop_binary),
            "uop_count": len(self.uops),
        }

    def save_to_directory(self, output_dir: str | Path) -> None:
        """Write artifact contents to disk."""

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # Separate files make it easy for testbenches to consume only what they need.
        (out_path / "uops.bin").write_bytes(self.uop_binary)
        (out_path / "params.bin").write_bytes(self.param_block)
        (out_path / "metadata.json").write_text(json.dumps(self.to_dict(), indent=2))
        # Optional binary Plan export (v1). Best-effort; does not break normal export.
        try:
            from venuscore_compiler.plan.binary_format import encode_plan_v1

            plan = self.metadata.get("plan") if isinstance(self.metadata, dict) else None
            if isinstance(plan, dict):
                (out_path / "plan.bin").write_bytes(encode_plan_v1(plan))
        except Exception:
            pass
        # Also emit a C header for firmware embedding by default.
        try:
            from venuscore_compiler.runtime.soc_exporter import export_c_header_arrays

            export_c_header_arrays(self, out_path / "bundle.h")
        except Exception:
            # Do not break normal export if header generation fails; this is a best-effort convenience.
            pass
