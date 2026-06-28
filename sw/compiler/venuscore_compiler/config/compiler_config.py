# -*- coding: utf-8 -*-
"""
Compiler configuration switches and policy knobs.

This module defines :class:`CompilerConfig`, which captures non-functional
behavior of the VenusCore compiler pipeline, such as whether to dump
intermediate IR or enable optional optimization passes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompilerConfig:
    """Compilation behavior toggles.

    These flags are intentionally kept coarse-grained. If in the future you
    need more fine-grained control (per-pass enable/disable, per-dialect
    dumping, etc.), extend this dataclass instead of adding ad-hoc globals.
    """

    # If True, dump midend IR / tiling plans / memory plans for debugging.
    dump_midend_debug: bool = False

    # If True, enable optional optimization passes in midend/backend.
    enable_opt_passes: bool = False

    # If True, enforce stricter assertion / consistency checks throughout the
    # pipeline (shape invariants, capacity checks, etc.). This is useful in
    # development and CI, but might be disabled in very constrained builds.
    strict_checks: bool = True
