# -*- coding: utf-8 -*-
"""
VenusCore functional simulator (uOP-level).

What this is:
- A functional (not cycle-accurate) model that executes VenusCore 32B uOPs (W0..W7)
- Matches: NCHWc4 memory layout, Param Block (qcoeff + align + weights),
  FIRST/LAST Cin-splitting accumulation, and SFU math per current RTL behavior.

Assumptions (match your current RTL style):
- Data layout: NCHWc4, one 32-bit word per pixel per c4-plane, little-endian bytes:
    byte0=lane0(ch%4==0), byte1=lane1, byte2=lane2, byte3=lane3
- QCoeff per output-channel: 8 bytes
    word0: bias (SInt32)
    word1: scale[15:0] (UInt16), shift[21:16] (UInt6), others ignored
- WeightBase = align16(param_addr + cout_tile*8)
- AVGPOOL/MAXPOOL are 2x2, stride fixed 2 (as per RTL)
- RELU6 behaves like RELU (current RTL clamp at 0 only, no cap at 6)

You can extend:
- MATMUL: define kernel/stride/K mapping and implement execute_*()

"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class SimConfig:
    word_bytes: int = 4
    uop_bytes: int = 32

    q_bytes_per_oc: int = 8
    param_align_bytes: int = 16

    strict_stride_encoding: bool = True
    relu6_is_relu: bool = True   # match RTL (True)
    # Match RTL: symmetric round-to-nearest before right shift in SFU, with ties
    # rounded away-from-zero (a.k.a. "half-up by magnitude").
    sfu_round_to_nearest: bool = True


# =============================================================================
# Sparse word memory (address space may be huge, store only touched words)
# =============================================================================

def _u32(x: int) -> int:
    return x & 0xFFFF_FFFF


class SparseWordMemory:
    """
    Stores 32-bit words at aligned addresses. Provides byte reads/writes.
    Uninitialized memory reads as 0.
    """
    def __init__(self) -> None:
        self._words: Dict[int, int] = {}

    def read_u32(self, addr: int) -> int:
        wa = addr & ~0x3
        return self._words.get(wa, 0)

    def write_u32(self, addr: int, value: int) -> None:
        wa = addr & ~0x3
        self._words[wa] = _u32(value)

    def read_u8(self, addr: int) -> int:
        wa = addr & ~0x3
        w = self._words.get(wa, 0)
        shift = (addr & 0x3) * 8
        return (w >> shift) & 0xFF

    def write_u8(self, addr: int, value: int) -> None:
        wa = addr & ~0x3
        w = self._words.get(wa, 0)
        shift = (addr & 0x3) * 8
        mask = 0xFF << shift
        w = (w & ~mask) | ((value & 0xFF) << shift)
        self._words[wa] = _u32(w)

    def load_bytes(self, base_addr: int, data: bytes) -> None:
        for i, b in enumerate(data):
            self.write_u8(base_addr + i, b)

    def load_bin(self, base_addr: int, path: str) -> int:
        with open(path, "rb") as f:
            data = f.read()
        self.load_bytes(base_addr, data)
        return len(data)

    def dump_bytes(self, base_addr: int, size: int) -> bytes:
        return bytes(self.read_u8(base_addr + i) for i in range(size))


# =============================================================================
# uOP decode
# =============================================================================

class Opcode(IntEnum):
    NOP     = 0x0
    CONV3x3 = 0x1
    PW1x1   = 0x2
    DW3x3   = 0x3
    AVGPOOL = 0x4
    MAXPOOL = 0x5
    MATMUL  = 0x6


class QMode(IntEnum):
    Q8 = 0
    Q4 = 1
    Q2 = 2


def _div4_ties_even_signed(acc: int) -> int:
    sign = -1 if acc < 0 else 1
    mag = -acc if acc < 0 else acc
    quo = mag >> 2
    rem = mag & 0x3
    inc = 1 if (rem > 2 or (rem == 2 and (quo & 0x1) == 1)) else 0
    return sign * (quo + inc)


@dataclass(frozen=True)
class Uop:
    # W0
    opcode: int
    act_type: int      # 0=none,1=relu,2=relu6
    first: int
    last: int
    stride_enc: int    # 1->stride1, 2->stride2
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    h_tile: int
    w_tile: int
    sync: int

    # W1
    c4_in: int
    c4_out: int
    y_index: int
    qmode: int

    # W2
    fi_stride: int     # words per input c4-plane
    fo_stride: int     # words per output c4-plane

    # W3..W5
    param_addr: int
    fi_addr: int
    fo_addr: int

    # W6
    ifm_w: int
    ifm_h: int

    # W7
    actdma_line_words: int
    outdma_line_words: int

    @staticmethod
    def _get_bits(x: int, hi: int, lo: int) -> int:
        mask = (1 << (hi - lo + 1)) - 1
        return (x >> lo) & mask

    @classmethod
    def decode_words(
        cls, w0: int, w1: int, w2: int, w3: int, w4: int, w5: int, w6: int, w7: int
    ) -> "Uop":
        # Bitfields per sw/compiler/doc/VenusCore_ISA.md (32B uOP, H+Y version).
        opcode    = cls._get_bits(w0,  3,  0)
        act_type  = cls._get_bits(w0,  6,  4)
        first     = cls._get_bits(w0,  7,  7)
        last      = cls._get_bits(w0,  8,  8)
        stride    = cls._get_bits(w0, 10,  9)
        pad_top   = cls._get_bits(w0, 11, 11)
        pad_bot   = cls._get_bits(w0, 12, 12)
        pad_left  = cls._get_bits(w0, 13, 13)
        pad_right = cls._get_bits(w0, 14, 14)
        h_tile    = cls._get_bits(w0, 22, 15)
        w_tile    = cls._get_bits(w0, 30, 23)
        sync      = cls._get_bits(w0, 31, 31)

        # W1 layout:
        #   9:0   C4_IN
        #   19:10 C4_OUT
        #   29:20 Y_INDEX
        #   31:30 QMODE
        c4_in     = cls._get_bits(w1,  9,  0)
        c4_out    = cls._get_bits(w1, 19, 10)
        y_index   = cls._get_bits(w1, 29, 20)
        qmode     = cls._get_bits(w1, 31, 30)

        fi_stride = cls._get_bits(w2, 15,  0)
        fo_stride = cls._get_bits(w2, 31, 16)

        return cls(
            opcode=opcode, act_type=act_type, first=first, last=last, stride_enc=stride,
            pad_top=pad_top, pad_bottom=pad_bot, pad_left=pad_left, pad_right=pad_right,
            h_tile=h_tile, w_tile=w_tile, sync=sync,
            c4_in=c4_in, c4_out=c4_out, y_index=y_index, qmode=qmode,
            fi_stride=fi_stride, fo_stride=fo_stride,
            param_addr=w3 & 0xFFFF_FFFF, fi_addr=w4 & 0xFFFF_FFFF, fo_addr=w5 & 0xFFFF_FFFF,
            ifm_w=cls._get_bits(w6, 15, 0),
            ifm_h=cls._get_bits(w6, 31, 16),
            actdma_line_words=cls._get_bits(w7, 15, 0),
            outdma_line_words=cls._get_bits(w7, 31, 16),
        )

    @classmethod
    def decode_bytes(cls, b: bytes, offset: int = 0) -> "Uop":
        if len(b) < offset + 32:
            raise ValueError("Not enough bytes for one uOP")
        ws = [int.from_bytes(b[offset + 4*i: offset + 4*i + 4], "little", signed=False) for i in range(8)]
        return cls.decode_words(*ws)

    @classmethod
    def decode_uops_bin(cls, path: str) -> List["Uop"]:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) % 32 != 0:
            raise ValueError(f"uops.bin size {len(data)} is not multiple of 32")
        return [cls.decode_bytes(data, off) for off in range(0, len(data), 32)]

    def stride(self) -> int:
        return int(self.stride_enc)

    def pads(self) -> Tuple[int, int, int, int]:
        return int(self.pad_top), int(self.pad_bottom), int(self.pad_left), int(self.pad_right)


# =============================================================================
# SFU + helpers
# =============================================================================

@dataclass(frozen=True)
class QCoeff:
    bias: int   # SInt32
    scale: int  # UInt16 (as used in your RTL)
    shift: int  # UInt6


def clamp_int8(x: int) -> int:
    return 127 if x > 127 else (-128 if x < -128 else int(x))


def apply_sfu(
    acc: int,
    q: QCoeff,
    act_type: int,
    relu6_is_relu: bool = True,
    round_to_nearest: bool = False,
) -> int:
    s = int(acc) + int(q.bias)
    if act_type in (1, 2):  # RELU / RELU6
        if s < 0:
            s = 0
        # current RTL: RELU6 == RELU
        if act_type == 2 and (not relu6_is_relu):
            s = min(s, 6)  # placeholder only
    prod = int(s) * int(q.scale)
    sh = int(q.shift)
    if round_to_nearest and sh > 0:
        # Symmetric round-to-nearest with ties away-from-zero:
        #   round(prod / 2**sh) where exact .5 goes to +/-1 away from zero.
        sign = -1 if prod < 0 else 1
        abs_p = -prod if prod < 0 else prod
        half = 1 << (sh - 1)
        res = sign * ((abs_p + half) >> sh)
    else:
        res = prod >> sh  # arithmetic shift
    return clamp_int8(res)


def unpack_i8x4_le(word_u32: int) -> Tuple[int, int, int, int]:
    b0 = (word_u32 >> 0) & 0xFF
    b1 = (word_u32 >> 8) & 0xFF
    b2 = (word_u32 >> 16) & 0xFF
    b3 = (word_u32 >> 24) & 0xFF

    def s8(b: int) -> int:
        return b - 256 if b >= 128 else b

    return s8(b0), s8(b1), s8(b2), s8(b3)


def pack_u8x4_le(b0: int, b1: int, b2: int, b3: int) -> int:
    return ((b0 & 0xFF) << 0) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24)


def align_up(x: int, align: int) -> int:
    return (x + (align - 1)) & ~(align - 1)


@dataclass(frozen=True)
class DerivedDims:
    ifm_w: int
    ifm_h: int
    ofm_w: int
    ofm_h: int


def derive_dims_from_uop(
    u: Uop,
    *,
    pads_override: Optional[Tuple[int, int, int, int]] = None,
) -> DerivedDims:
    """
    Derive IFM/OFM spatial dimensions from ISA fields.

    ISA provides:
      - W_TILE (OFM width, no W tiling)
      - FI_STRIDE = IFM_H * IFM_W (words)
      - FO_STRIDE = OFM_H * OFM_W (words)
      - stride + padding + opcode (implies kernel size)
      - IFM_W / IFM_H (explicit in W6)

    当前版本优先使用 W6 中显式提供的 IFM_W/IFM_H，避免从 FI_STRIDE/FO_STRIDE 反推，
    从而避免除法或候选枚举。

    注意：
      uOP 里的 pad_* 是“该 tile 是否处在边界需要 padding”的 1-bit 标志，
      并不是全局 padding 量。比如做 H 方向分 tile 时：
        - 顶部 tile: pad_top=1, pad_bottom=0
        - 中间 tile: pad_top=0, pad_bottom=0
        - 底部 tile: pad_top=0, pad_bottom=1
      用单条 uOP 的 pad 去反推全局 IFM_H/IFM_W 会失败。
      因此本函数支持 pads_override：由上层先把同一层/同一组参数的所有 uOP
      的 pad_* 做 max 聚合得到“全局 pad”，再传进来推导维度。
    """
    if pads_override is None:
        pad_top, pad_bot, pad_left, pad_right = u.pads()
    else:
        pad_top, pad_bot, pad_left, pad_right = (int(x) for x in pads_override)
    stride = u.stride()

    ofm_w = u.w_tile
    if ofm_w <= 0:
        raise ValueError(f"Derived OFM_W<=0: {ofm_w}")
    if u.fo_stride % ofm_w != 0:
        raise ValueError(f"FO_STRIDE({u.fo_stride}) not divisible by OFM_W({ofm_w})")
    ofm_h = u.fo_stride // ofm_w

    # Determine kernel sizes implied by opcode (current RTL subset).
    if u.opcode in (Opcode.AVGPOOL, Opcode.MAXPOOL):
        kernel_h = kernel_w = 2
    elif u.opcode in (Opcode.CONV3x3, Opcode.DW3x3):
        kernel_h = kernel_w = 3
    else:
        kernel_h = kernel_w = 1

    ifm_w = int(u.ifm_w)
    ifm_h = int(u.ifm_h)
    if ifm_w <= 0 or ifm_h <= 0:
        raise ValueError(f"Invalid IFM dims from W6: IFM_H={ifm_h} IFM_W={ifm_w}")
    if int(u.fi_stride) != (ifm_h * ifm_w):
        raise ValueError(
            f"FI_STRIDE mismatch: fi_stride={u.fi_stride} vs ifm_h*ifm_w={ifm_h * ifm_w} (IFM={ifm_h}x{ifm_w})"
        )

    # 额外一致性检查（调试友好）：IFM/OFM 与 stride/pad/kernel 是否匹配输出宽高。
    def _out_size(in_size: int, pad0: int, pad1: int, k: int, s: int) -> int:
        num = in_size + pad0 + pad1 - k
        if num < 0:
            return -1
        return (num // s) + 1

    if _out_size(ifm_w, pad_left, pad_right, kernel_w, stride) != ofm_w:
        raise ValueError(
            f"OFM_W mismatch: expect {ofm_w}, got {_out_size(ifm_w, pad_left, pad_right, kernel_w, stride)} "
            f"(IFM_W={ifm_w}, pads L/R={pad_left}/{pad_right}, K_w={kernel_w}, stride={stride})"
        )
    if _out_size(ifm_h, pad_top, pad_bot, kernel_h, stride) != ofm_h:
        raise ValueError(
            f"OFM_H mismatch: expect {ofm_h}, got {_out_size(ifm_h, pad_top, pad_bot, kernel_h, stride)} "
            f"(IFM_H={ifm_h}, pads T/B={pad_top}/{pad_bot}, K_h={kernel_h}, stride={stride})"
        )

    return DerivedDims(ifm_w=ifm_w, ifm_h=ifm_h, ofm_w=ofm_w, ofm_h=ofm_h)


def load_qcoeff(mem: SparseWordMemory, param_addr: int, cout_tile: int) -> List[QCoeff]:
    qs: List[QCoeff] = []
    for oc in range(cout_tile):
        base = param_addr + oc * 8
        bias_u32 = mem.read_u32(base + 0)
        bias = bias_u32 - 0x1_0000_0000 if (bias_u32 & 0x8000_0000) else bias_u32
        hi = mem.read_u32(base + 4)
        scale = hi & 0xFFFF
        shift = (hi >> 16) & 0x3F
        qs.append(QCoeff(bias=bias, scale=scale, shift=shift))
    return qs


def weight_base(param_addr: int, cout_tile: int, q_bytes_per_oc: int, align_bytes: int) -> int:
    q_size = cout_tile * q_bytes_per_oc
    return align_up(param_addr + q_size, align_bytes)


# =============================================================================
# Simulator core
# =============================================================================

def _acc_key(u: Uop) -> Tuple[int, int, int, int, int]:
    # Key for Cin-split accumulation across multiple uOPs
    return (u.fo_addr, u.y_index, u.h_tile, u.w_tile, u.c4_out)


class VenusCoreSim:
    def __init__(self, cfg: SimConfig, mem: SparseWordMemory) -> None:
        self.cfg = cfg
        self.mem = mem
        self._acc: Dict[Tuple[int, int, int, int, int], List[int]] = {}
        self._layer_pads: Dict[Tuple, Tuple[int, int, int, int]] = {}

    def reset(self) -> None:
        self._acc.clear()
        self._layer_pads.clear()

    def prepare(self, uops: List[Uop]) -> None:
        self._precompute_layer_pads(uops)

    @staticmethod
    def _layer_key(u: Uop) -> Tuple:
        # 以“同一组权重/参数 + 同一层几何参数”为 key，把分 tile 的 pad 标志聚合成全局 pad。
        # param_addr 对于同一层的不同 y tile 通常相同（权重共享），因此可作为主键的一部分。
        return (
            int(u.opcode),
            int(u.param_addr),
            int(u.fi_stride),
            int(u.fo_stride),
            int(u.w_tile),
            int(u.c4_in),
            int(u.c4_out),
            int(u.stride()),
        )

    def _precompute_layer_pads(self, uops: List[Uop]) -> None:
        self._layer_pads.clear()
        for u in uops:
            k = self._layer_key(u)
            pt, pb, pl, pr = u.pads()
            if k not in self._layer_pads:
                self._layer_pads[k] = (pt, pb, pl, pr)
            else:
                opt, opb, opl, opr = self._layer_pads[k]
                self._layer_pads[k] = (
                    max(opt, pt),
                    max(opb, pb),
                    max(opl, pl),
                    max(opr, pr),
                )

    def run(self, uops: List[Uop], verbose: bool = False, max_uops: Optional[int] = None) -> None:
        self.prepare(uops)
        n = len(uops) if max_uops is None else min(len(uops), max_uops)
        for i in range(n):
            self.exec_one(uops[i], verbose=verbose)

    def exec_one(self, u: Uop, verbose: bool = False) -> None:
        if u.qmode != QMode.Q8:
            is_pool_ties_even = (u.opcode == Opcode.AVGPOOL) and (u.qmode == QMode.Q4)
            if not is_pool_ties_even:
                raise NotImplementedError(f"Unsupported QMODE/opcode combination: qmode={u.qmode} opcode={u.opcode}")

        stride = u.stride()
        if self.cfg.strict_stride_encoding and stride not in (1, 2):
            raise ValueError(f"Illegal STRIDE encoding: {u.stride_enc}")

        pads = self._layer_pads.get(self._layer_key(u))
        dims = derive_dims_from_uop(u, pads_override=pads)

        cout_tile = u.c4_out * 4
        qcoeff = load_qcoeff(self.mem, u.param_addr, cout_tile)

        # NOTE:
        # FIRST_FLAG / LAST_FLAG are stream-level markers: the first/last uOP of
        # a submitted uOP stream (typically one NPU subgraph / one driver submit).
        #
        # This simulator currently ignores FIRST/LAST for functional behavior;
        # it always writes back each uOP's outputs.
        acc_buf = [0] * (cout_tile * u.h_tile * u.w_tile)
        do_clear = True
        do_writeback = True

        if verbose:
            print(
                f"[SIM] op={u.opcode} act={u.act_type} first={u.first} last={u.last} stride={u.stride_enc} "
                f"pad={u.pads()} HxW={u.h_tile}x{u.w_tile} y_index={u.y_index} "
                f"c4_in={u.c4_in} c4_out={u.c4_out} "
                f"IFM={dims.ifm_h}x{dims.ifm_w} OFM={dims.ofm_h}x{dims.ofm_w} "
                f"PARAM=0x{u.param_addr:08X} FI=0x{u.fi_addr:08X} FO=0x{u.fo_addr:08X}"
            )

        op = Opcode(u.opcode)

        if op == Opcode.NOP:
            return

        if op == Opcode.CONV3x3:
            self._exec_conv_or_pw(u, dims, qcoeff, acc_buf, do_clear, do_writeback, is_pw=False, global_pads=pads)
        elif op == Opcode.PW1x1:
            self._exec_conv_or_pw(u, dims, qcoeff, acc_buf, do_clear, do_writeback, is_pw=True, global_pads=pads)
        elif op == Opcode.DW3x3:
            self._exec_dw3x3(u, dims, qcoeff, acc_buf, do_clear, do_writeback, global_pads=pads)
        elif op == Opcode.AVGPOOL:
            self._exec_avgpool2x2(u, dims, qcoeff, global_pads=pads)
            # No accumulator state is kept across uOPs.
        elif op == Opcode.MAXPOOL:
            self._exec_maxpool2x2(u, dims, qcoeff, global_pads=pads)
        elif op == Opcode.MATMUL:
            raise NotImplementedError("MATMUL not implemented yet")
        else:
            raise NotImplementedError(f"Unsupported opcode: {u.opcode}")
        # No accumulator state is kept across uOPs.

    # ------------------------------
    # Operators
    # ------------------------------

    def _exec_conv_or_pw(
        self,
        u: Uop,
        dims: DerivedDims,
        qcoeff: List[QCoeff],
        acc_buf: List[int],
        do_clear: bool,
        do_writeback: bool,
        is_pw: bool,
        global_pads: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        c4_in = u.c4_in
        c4_out = u.c4_out
        cout_tile = c4_out * 4

        h_tile = u.h_tile
        w_tile = u.w_tile
        y_base = u.y_index

        stride = u.stride()
        if global_pads is None:
            pad_top, pad_bot, pad_left, pad_right = u.pads()
        else:
            pad_top, pad_bot, pad_left, pad_right = (int(x) for x in global_pads)

        if is_pw:
            kh_list = (0,)
            kw_list = (0,)
            taps = 1
        else:
            kh_list = (0, 1, 2)
            kw_list = (0, 1, 2)
            taps = 9

        wb = weight_base(u.param_addr, cout_tile, self.cfg.q_bytes_per_oc, self.cfg.param_align_bytes)

        if do_clear:
            for i in range(len(acc_buf)):
                acc_buf[i] = 0

        # Preload weights into [oc][c4*taps + tap]
        weights: List[List[int]] = [[0] * (c4_in * taps) for _ in range(cout_tile)]
        for oc in range(cout_tile):
            base_oc = wb + (oc * c4_in * taps) * 4
            wi = 0
            for c4 in range(c4_in):
                for kh in kh_list:
                    for kw in kw_list:
                        tap = (kh * 3 + kw) if not is_pw else 0
                        w_addr = base_oc + (c4 * taps + tap) * 4
                        weights[oc][wi] = self.mem.read_u32(w_addr)
                        wi += 1

        # MAC accumulate
        for oy in range(h_tile):
            gy = y_base + oy
            if gy < 0 or gy >= dims.ofm_h:
                continue
            for ox in range(w_tile):
                gx = ox  # ISA convention: W_TILE equals the global W_out (no W tiling)
                for oc in range(cout_tile):
                    idx = ((oc * h_tile) + oy) * w_tile + ox
                    acc = acc_buf[idx]
                    wi = 0
                    for c4 in range(c4_in):
                        for kh in kh_list:
                            for kw in kw_list:
                                iy = gy * stride + kh - pad_top
                                ix = gx * stride + kw - pad_left
                                if ix < 0 or ix >= dims.ifm_w or iy < 0 or iy >= dims.ifm_h:
                                    wi += 1
                                    continue
                                in_word = self.mem.read_u32(u.fi_addr + (c4 * u.fi_stride + iy * dims.ifm_w + ix) * 4)
                                a0, a1, a2, a3 = unpack_i8x4_le(in_word)

                                w_word = weights[oc][wi]
                                w0, w1, w2, w3 = unpack_i8x4_le(w_word)

                                acc += a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3
                                wi += 1
                    acc_buf[idx] = acc

        if not do_writeback:
            return

        # SFU + write OFM
        for oy in range(h_tile):
            gy = y_base + oy
            if gy < 0 or gy >= dims.ofm_h:
                continue
            for ox in range(w_tile):
                gx = ox
                for g in range(c4_out):
                    outb = [0, 0, 0, 0]
                    for lane in range(4):
                        oc = g * 4 + lane
                        idx = ((oc * h_tile) + oy) * w_tile + ox
                        out_i8 = apply_sfu(
                            acc_buf[idx],
                            qcoeff[oc],
                            u.act_type,
                            relu6_is_relu=self.cfg.relu6_is_relu,
                            round_to_nearest=self.cfg.sfu_round_to_nearest,
                        )
                        outb[lane] = out_i8 & 0xFF
                    out_word = pack_u8x4_le(outb[0], outb[1], outb[2], outb[3])
                    out_addr = u.fo_addr + (g * u.fo_stride + gy * dims.ofm_w + gx) * 4
                    self.mem.write_u32(out_addr, out_word)

    def _exec_dw3x3(
        self,
        u: Uop,
        dims: DerivedDims,
        qcoeff: List[QCoeff],
        acc_buf: List[int],
        do_clear: bool,
        do_writeback: bool,
        global_pads: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        c4_out = u.c4_out
        cout_tile = c4_out * 4

        h_tile = u.h_tile
        w_tile = u.w_tile
        y_base = u.y_index

        stride = u.stride()
        if global_pads is None:
            pad_top, pad_bot, pad_left, pad_right = u.pads()
        else:
            pad_top, pad_bot, pad_left, pad_right = (int(x) for x in global_pads)

        wb = weight_base(u.param_addr, cout_tile, self.cfg.q_bytes_per_oc, self.cfg.param_align_bytes)

        if do_clear:
            for i in range(len(acc_buf)):
                acc_buf[i] = 0

        # DW weight: per channel, 3 words (kw=0..2), word packs {top,mid,bot,0}
        w_dw: List[List[int]] = [[0, 0, 0] for _ in range(cout_tile)]
        for ch in range(cout_tile):
            base_ch = wb + ch * 3 * 4
            for kw in range(3):
                w_dw[ch][kw] = self.mem.read_u32(base_ch + kw * 4)

        for oy in range(h_tile):
            gy = y_base + oy
            if gy < 0 or gy >= dims.ofm_h:
                continue
            for ox in range(w_tile):
                gx = ox
                for ch in range(cout_tile):
                    idx = ((ch * h_tile) + oy) * w_tile + ox
                    acc = acc_buf[idx]
                    g = ch // 4
                    lane = ch % 4

                    for kw in range(3):
                        ix = gx * stride + kw - pad_left
                        if ix < 0 or ix >= dims.ifm_w:
                            continue

                        wt, wm, wbw, _ = unpack_i8x4_le(w_dw[ch][kw])
                        for kh, wv in enumerate((wt, wm, wbw)):
                            iy = gy * stride + kh - pad_top
                            if iy < 0 or iy >= dims.ifm_h:
                                continue
                            in_word = self.mem.read_u32(u.fi_addr + (g * u.fi_stride + iy * dims.ifm_w + ix) * 4)
                            a0, a1, a2, a3 = unpack_i8x4_le(in_word)
                            aval = (a0, a1, a2, a3)[lane]
                            acc += aval * wv

                    acc_buf[idx] = acc

        if not do_writeback:
            return

        for oy in range(h_tile):
            gy = y_base + oy
            if gy < 0 or gy >= dims.ofm_h:
                continue
            for ox in range(w_tile):
                gx = ox
                for g in range(c4_out):
                    outb = [0, 0, 0, 0]
                    for lane in range(4):
                        ch = g * 4 + lane
                        idx = ((ch * h_tile) + oy) * w_tile + ox
                        out_i8 = apply_sfu(
                            acc_buf[idx],
                            qcoeff[ch],
                            u.act_type,
                            relu6_is_relu=self.cfg.relu6_is_relu,
                            round_to_nearest=self.cfg.sfu_round_to_nearest,
                        )
                        outb[lane] = out_i8 & 0xFF
                    out_word = pack_u8x4_le(outb[0], outb[1], outb[2], outb[3])
                    out_addr = u.fo_addr + (g * u.fo_stride + gy * dims.ofm_w + gx) * 4
                    self.mem.write_u32(out_addr, out_word)

    def _exec_avgpool2x2(
        self,
        u: Uop,
        dims: DerivedDims,
        qcoeff: List[QCoeff],
        global_pads: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        # current RTL: AVG2, stride fixed 2
        if u.c4_out != u.c4_in:
            raise ValueError(f"AVGPOOL expects C4_OUT==C4_IN, got {u.c4_out} vs {u.c4_in}")

        c4 = u.c4_out
        cout_tile = c4 * 4

        h_tile = u.h_tile
        w_tile = u.w_tile
        y_base = u.y_index

        if global_pads is None:
            pad_top, pad_bot, pad_left, pad_right = u.pads()
        else:
            pad_top, pad_bot, pad_left, pad_right = (int(x) for x in global_pads)
        stride = 2

        for oy in range(h_tile):
            gy = y_base + oy
            if gy < 0 or gy >= dims.ofm_h:
                continue
            for ox in range(w_tile):
                gx = ox
                for g in range(c4):
                    outb = [0, 0, 0, 0]
                    for lane in range(4):
                        ch = g * 4 + lane
                        acc = 0
                        for kh in (0, 1):
                            for kw in (0, 1):
                                iy = gy * stride + kh - pad_top
                                ix = gx * stride + kw - pad_left
                                if ix < 0 or ix >= dims.ifm_w or iy < 0 or iy >= dims.ifm_h:
                                    continue
                                in_word = self.mem.read_u32(u.fi_addr + (g * u.fi_stride + iy * dims.ifm_w + ix) * 4)
                                a0, a1, a2, a3 = unpack_i8x4_le(in_word)
                                acc += (a0, a1, a2, a3)[lane]

                        # Default AVGPOOL semantic remains arithmetic truncation.
                        # For selected KWS layers we reuse QMODE=Q4 as an opt-in
                        # marker for ties-to-even divide-by-4 rounding.
                        if u.qmode == int(QMode.Q4):
                            acc = _div4_ties_even_signed(acc)
                        else:
                            acc = acc >> 2
                        out_i8 = apply_sfu(
                            acc,
                            qcoeff[ch],
                            u.act_type,
                            relu6_is_relu=self.cfg.relu6_is_relu,
                            round_to_nearest=self.cfg.sfu_round_to_nearest,
                        )
                        outb[lane] = out_i8 & 0xFF

                    out_word = pack_u8x4_le(outb[0], outb[1], outb[2], outb[3])
                    out_addr = u.fo_addr + (g * u.fo_stride + gy * dims.ofm_w + gx) * 4
                    self.mem.write_u32(out_addr, out_word)

    def _exec_maxpool2x2(
        self,
        u: Uop,
        dims: DerivedDims,
        qcoeff: List[QCoeff],
        global_pads: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        if u.c4_out != u.c4_in:
            raise ValueError(f"MAXPOOL expects C4_OUT==C4_IN, got {u.c4_out} vs {u.c4_in}")

        c4 = u.c4_out
        h_tile = u.h_tile
        w_tile = u.w_tile
        y_base = u.y_index

        if global_pads is None:
            pad_top, pad_bot, pad_left, pad_right = u.pads()
        else:
            pad_top, pad_bot, pad_left, pad_right = (int(x) for x in global_pads)
        stride = 2
        pad_val = -128

        for oy in range(h_tile):
            gy = y_base + oy
            if gy < 0 or gy >= dims.ofm_h:
                continue
            for ox in range(w_tile):
                gx = ox
                for g in range(c4):
                    outb = [0, 0, 0, 0]
                    for lane in range(4):
                        ch = g * 4 + lane
                        max_val = pad_val
                        for kh in (0, 1):
                            for kw in (0, 1):
                                iy = gy * stride + kh - pad_top
                                ix = gx * stride + kw - pad_left
                                if ix < 0 or ix >= dims.ifm_w or iy < 0 or iy >= dims.ifm_h:
                                    v = pad_val
                                else:
                                    in_word = self.mem.read_u32(
                                        u.fi_addr + (g * u.fi_stride + iy * dims.ifm_w + ix) * 4
                                    )
                                    a0, a1, a2, a3 = unpack_i8x4_le(in_word)
                                    v = (a0, a1, a2, a3)[lane]
                                if v > max_val:
                                    max_val = v

                        out_i8 = apply_sfu(
                            max_val,
                            qcoeff[ch],
                            u.act_type,
                            relu6_is_relu=self.cfg.relu6_is_relu,
                            round_to_nearest=self.cfg.sfu_round_to_nearest,
                        )
                        outb[lane] = out_i8 & 0xFF

                    out_word = pack_u8x4_le(outb[0], outb[1], outb[2], outb[3])
                    out_addr = u.fo_addr + (g * u.fo_stride + gy * dims.ofm_w + gx) * 4
                    self.mem.write_u32(out_addr, out_word)


# =============================================================================
# Minimal CLI-like helpers
# =============================================================================

def load_uops(path: str) -> List[Uop]:
    return Uop.decode_uops_bin(path)


def main_example():
    """
    Example (adjust addresses to your TB convention):
      mem.load_bin(PARAM_ADDR, "params.bin")
      mem.load_bin(UOP_ADDR, "uops.bin")   # optional, we decode from file anyway
      load IFM region into mem at FI_ADDR
      run uops
      dump OFM bytes from FO_ADDR
    """
    mem = SparseWordMemory()
    cfg = SimConfig()
    sim = VenusCoreSim(cfg, mem)

    uops = load_uops("uops.bin")

    # If your params.bin is expected to be at the same base as uop.W3, load there:
    # e.g. PARAM_ADDR = uops[0].param_addr
    if uops:
        mem.load_bin(uops[0].param_addr, "params.bin")

    # TODO: load IFM bytes to uops[0].fi_addr ...

    sim.run(uops, verbose=True)
    print("[SIM] done.")


if __name__ == "__main__":
    # main_example()
    pass
