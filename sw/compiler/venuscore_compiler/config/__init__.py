# -*- coding: utf-8 -*-
"""Public configuration API for the VenusCore compiler.

This package exposes:

- HwConfig:       hardware resource / capability description
- CompilerConfig: compiler behavior knobs
- default_hw_config(target):   default HwConfig for common targets
- default_compiler_config():   conservative compiler defaults

Typical usage:

    from venuscore_compiler.config import (
        HwConfig,
        CompilerConfig,
        default_hw_config,
        default_compiler_config,
    )

    hw_cfg = default_hw_config("zybo7010")
    cc_cfg = default_compiler_config()
"""

from .hw_config import HwConfig
from .compiler_config import CompilerConfig
from .defaults import default_hw_config, default_compiler_config

__all__ = [
    "HwConfig",
    "CompilerConfig",
    "default_hw_config",
    "default_compiler_config",
]
