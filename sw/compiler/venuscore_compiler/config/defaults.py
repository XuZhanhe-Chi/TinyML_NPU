# -*- coding: utf-8 -*-
"""Default hardware/compiler configs for the public TinyML_NPU targets."""

from __future__ import annotations

from venuscore_compiler.config.hw_config import HwConfig
from venuscore_compiler.config.compiler_config import CompilerConfig


def default_hw_config(target: str = "zybo7010") -> HwConfig:
    """Return a default :class:`HwConfig` for the given target."""

    if target in {"zybo7010", "default"}:
        return HwConfig(
            ibuf_line_bytes=3840,
            wbuf_lane_bytes=2048,
            wbuf_lanes=4,
            max_h_tile=255,
            cluster_count=1,
        )
    if target == "sim":
        return HwConfig(
            ibuf_line_bytes=3840,
            wbuf_lane_bytes=2048,
            wbuf_lanes=4,
            max_h_tile=255,
            cluster_count=1,
        )
    raise ValueError(f"Unknown hw target: {target!r}")


def default_compiler_config() -> CompilerConfig:
    """Return a :class:`CompilerConfig` instance with conservative defaults."""

    return CompilerConfig()
