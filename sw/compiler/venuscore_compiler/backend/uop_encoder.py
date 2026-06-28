# -*- coding: utf-8 -*-
"""
Wrapper for encoding semantic uOPs to bytes.
"""

from __future__ import annotations

from typing import Iterable

from venuscore_compiler.isa.encoder import encode_uops
from venuscore_compiler.isa.uop_format import Uop

__all__ = ["encode_uops_bytes"]


def encode_uops_bytes(uops: Iterable[Uop]) -> bytes:
    """Encode uOPs to binary using ISA encoder."""

    return encode_uops(list(uops))
