# -*- coding: utf-8 -*-
"""
Module overview:
  - ISA encoding/decoding/enums public API.
  - Dependencies:
    * Depends on: isa.encoder, isa.decoder, isa.layout_spec, isa.uop_format
    * Used by: backend and debug scripts
"""


from venuscore_compiler.isa.decoder import decode_uop, decode_uops
from venuscore_compiler.isa.encoder import encode_uop, encode_uops
from venuscore_compiler.isa.layout_spec import Activation, Opcode, QMode
from venuscore_compiler.isa.uop_format import UOP_SIZE_BYTES, Uop

__all__ = [
    "Activation",
    "Opcode",
    "QMode",
    "UOP_SIZE_BYTES",
    "Uop",
    "decode_uop",
    "decode_uops",
    "encode_uop",
    "encode_uops",
]
