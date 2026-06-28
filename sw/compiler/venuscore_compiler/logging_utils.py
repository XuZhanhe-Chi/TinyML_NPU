# -*- coding: utf-8 -*-
"""
Module overview:
  - Logging configuration and helpers.
  - Dependencies:
    * Depends on: Python stdlib logging
    * Used by: cli and internal modules for consistent logging
"""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Create and configure the package logger."""

    logger = logging.getLogger("venuscore_compiler")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(levelname)s:%(name)s:%(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level.upper())
    return logger
