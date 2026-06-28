# -*- coding: utf-8 -*-
"""
Naive linear memory planner for IFM/OFM regions and Param Block blob.

Module overview:
  - Defines :class:`MemoryPlan` and :func:`plan_memory` for a simple,
    hardware-agnostic memory layout.
  - Dependencies: midend.tiler.TilePlan / TileDesc / LayerConfig.
  - Scope: assigns fixed base regions; future revisions should align with
    hardware memory map and tile-aware planning.
  - Addressing:
        global_address(IFM[n,c,h,w]) =
            ifm_offsets[name] + compute_ifm_offset(...)
    similarly for OFM. Param Block is a contiguous blob placed at
    ``param_base`` with size ``param_size``, and per-tile PARAM_ADDR
    is tracked in ``param_offsets``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from venuscore_compiler.midend.types import TilePlan, TileDesc, LayerConfig
from venuscore_compiler.backend.layout_ifm_ofm import WORD_BYTES
from venuscore_compiler.ir.program import VcProgram

ALIGNMENT = 16


@dataclass
class MemoryPlan:
    """
    Records byte offsets for IFM/OFM bases and a single Param Block blob.

    Attributes:
        ifm_offsets:
            Mapping from logical IFM tensor name to global base address (bytes).

        ofm_offsets:
            Mapping from logical OFM tensor name to global base address (bytes).

        param_base:
            Global base address (bytes) where the contiguous Param Block blob
            starts. Individual tiles use PARAM_ADDR = param_offsets[(op, tile_id)].

        param_size:
            Total size in bytes of the Param Block blob.

        param_offsets:
            Mapping (op_name, tile_id) -> PARAM_ADDR (absolute byte address).
            The Param Block builder uses these offsets as "expected" positions
            and will pad or error if mismatched.

        total_size:
            Aggregate size (bytes) of IFM + OFM regions under this naive layout.
            Param Block size is tracked separately in param_size.

        layer_configs:
            Optional cached LayerConfig list for backend convenience
            (e.g. lookup by layer_id). This is not strictly required by the
            planner itself, but useful to keep the full context here.
    """

    ifm_offsets: Dict[str, int] = field(default_factory=dict)
    ofm_offsets: Dict[str, int] = field(default_factory=dict)

    param_base: int = 0
    param_size: int = 0
    param_offsets: Dict[Tuple[str, int], int] = field(default_factory=dict)

    total_size: int = 0
    layer_configs: Dict[int, LayerConfig] = field(default_factory=dict)

    # Peak activation footprint under current allocation strategy (bytes).
    activation_peak_bytes: int = 0

    # Filled once param_size is known; mirrors param_size for convenience.
    weight_bytes: int = 0


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def plan_memory(
    tile_plan: TilePlan,
    ifm_base: int = 0,
    ofm_base: int | None = None,
    param_base: int | None = None,
    allocation_mode: str = "ping_pong",
    program: VcProgram | None = None,
) -> MemoryPlan:
    """
    Plan a memory layout for IFM / OFM / Param Block.

    Layout strategy:
        - Activation region:
            * "ping_pong" (default): two reusable buffers sized to peak IFM/OFM.
              Layer0 reads A / writes B, layer1 reads B / writes A, ...
            * "per_tensor": legacy behavior, each IFM/OFM tensor gets its own
              contiguous block.
        - Param Block blob: starts at ``param_base`` (or right after activation
          region if None) with per-tile offsets.

    The actual Param Block bytes are later constructed by
    :mod:`backend.layout_param_block`, which will enforce that the real
    blob layout matches the offsets assigned here.

    Args:
        tile_plan:
            Tiling plan produced by midend, containing:
              - layers: List[LayerConfig]
              - tiles_by_op: Dict[str, List[TileDesc]]
        ifm_base:
            Global base address (bytes) used for the first IFM tensor.
        ofm_base:
            Global base address (bytes) used for the first OFM tensor. If None,
            OFM region starts right after the IFM region.
        param_base:
            Global base address (bytes) used for the Param Block blob. If None,
            it starts right after the OFM region.

    Returns:
        A populated :class:`MemoryPlan` instance.
    """
    mp = MemoryPlan()

    # Cache LayerConfig by layer_id for fast lookup.
    layer_cfgs = getattr(tile_plan, "layers", [])
    mp.layer_configs = {cfg.layer_id: cfg for cfg in layer_cfgs}

    # allocation_mode:
    #   - "ping_pong": two reusable activation buffers (works for strictly linear chains).
    #   - "arena": single address per logical tensor name (required for fan-out / CPU fallback).
    #   - "per_tensor": kept for backward compatibility; treated as "arena".
    if allocation_mode == "per_tensor":
        allocation_mode = "arena"

    if allocation_mode not in ("ping_pong", "arena"):
        raise ValueError(f"Unknown allocation_mode {allocation_mode!r}")

    if allocation_mode == "ping_pong":
        _assign_ping_pong(layer_cfgs, mp, ifm_base, ofm_base)
    else:
        # In plan/fallback mode we prefer a single arena (ofm_base=None) and a
        # liveness-based allocator to avoid allocating a unique buffer per tensor.
        # Fall back to the legacy "no reuse" arena layout when:
        #   - no program is provided, or
        #   - the caller forces a separate produced_base via ofm_base.
        if program is not None and ofm_base is None:
            _assign_arena_liveness(program, layer_cfgs, mp, ifm_base)
        else:
            _assign_arena(layer_cfgs, mp, ifm_base, ofm_base)

    # -------------------------------------------------------------------------
    # 3) Param Block blob: per-tile offsets (PARAM_ADDR) under a linear layout
    # -------------------------------------------------------------------------
    if param_base is None:
        param_base = _align_up(_activation_region_end(layer_cfgs, mp), ALIGNMENT)
    mp.param_base = param_base

    cursor = param_base

    # Reuse param offsets for tiles that share the same weight slice (same op and co_start/co_end).
    shared_offsets: Dict[Tuple[str, int, int], int] = {}

    for op_name, tiles in tile_plan.tiles_by_op.items():
        for tile in tiles:
            key = (op_name, tile.tile_id)
            share_key = (op_name, tile.co_start, tile.co_end)
            if share_key in shared_offsets:
                mp.param_offsets[key] = shared_offsets[share_key]
                continue

            mp.param_offsets[key] = cursor
            shared_offsets[share_key] = cursor

            layer_cfg = mp.layer_configs.get(tile.layer_id)
            est_size = _estimate_param_block_size(tile, layer_cfg)
            cursor += est_size

    mp.param_size = cursor - mp.param_base
    mp.weight_bytes = mp.param_size
    return mp


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _estimate_param_block_size(tile: TileDesc, cfg: LayerConfig | None) -> int:
    """
    Estimate Param Block size (bytes) for a tile.

    This estimation must stay aligned with backend.layout_param_block:

        size =
            QuantBytes (Cout_tile * 8) +
            alignment to 16B +
            WeightBytes(op_type-specific)

    For now, MATMUL/FC return 0 here, consistent with layout_param_block
    raising NotImplementedError for these op types.
    """
    co_start = tile.co_start
    co_end = tile.co_end
    cout_tile = co_end - co_start
    if cout_tile <= 0:
        return 0

    # 1) Quant Coeff Block
    quant_bytes = cout_tile * 8

    # 2) Alignment to 16B for WeightBase
    weight_base = (quant_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1)

    # 3) Weight Data Block (op_type-specific)
    op_type = tile.op_type.lower()
    kernel_bytes = 0

    if op_type in ("conv2d", "pointwise_conv"):
        # Weight tensor layout: [Cout_total, Cin_total, Kh, Kw]
        # Cin is not tiled; use LayerConfig.cin if available.
        cin_total = getattr(cfg, "cin", 0) if cfg is not None else getattr(tile, "cin", 0)
        kh = getattr(tile, "kernel_h", 1)
        kw = getattr(tile, "kernel_w", 1)

        c4_in = (cin_total + 3) // 4 if cin_total > 0 else 0
        kernel_bytes = cout_tile * c4_in * kh * kw * WORD_BYTES

    elif op_type in ("depthwise_conv", "dwconv"):
        # DWCONV 3x3:
        #   Cin_dw = Cin_tile (channel-wise slicing, aligned to C4)
        #   For each ch in [0..Cin_dw-1], 3 words (kw=0,1,2), each word:
        #       {top, mid, bot, 0}
        #   => bytes = Cin_dw * 3 * 4
        # For DWConv, Cout tiling implies the same channel slice on input:
        #   tile.cout == tile.cin == Cin_tile.
        cin_dw = int(getattr(tile, "cout", 0))
        if cin_dw > 0:
            kernel_bytes = cin_dw * 3 * WORD_BYTES
        else:
            kernel_bytes = 0

    elif op_type in ("fully_connected", "matmul", "matmul_fc"):
        # Keep 0 here; layout_param_block will raise NotImplementedError.
        kernel_bytes = 0
    elif op_type in ("avg_pool", "avgpool", "max_pool", "maxpool"):
        # Pool ops have no weights, but still consume Quant Coeff Block
        # (bias/scale/shift) for SFU requantization.
        kernel_bytes = 0

    total = weight_base + kernel_bytes
    return total if total > 0 else 0


def _assign_ping_pong(
    layer_cfgs: list[LayerConfig],
    mp: MemoryPlan,
    ifm_base: int,
    ofm_base: int | None,
) -> None:
    """
    Assign two reusable activation buffers sized to peak IFM/OFM demand.

    Layer k reads from buffer A and writes to buffer B if k is even, and vice versa if k is odd.
    """
    max_ifm = 0
    max_ofm = 0
    for cfg in layer_cfgs:
        max_ifm = max(max_ifm, _activation_bytes_from_cfg(cfg, is_ifm=True))
        max_ofm = max(max_ofm, _activation_bytes_from_cfg(cfg, is_ifm=False))

    # Keep 16B alignment for all activation bases to match Plan/driver assumptions.
    buf_a_base = _align_up(int(ifm_base), ALIGNMENT)
    buf_a_size = _align_up(int(max_ifm), ALIGNMENT)
    buf_b_base = _align_up(int(ofm_base), ALIGNMENT) if ofm_base is not None else buf_a_base + buf_a_size
    buf_b_size = _align_up(int(max_ofm), ALIGNMENT)

    # Ensure buffers do not overlap.
    end_a = buf_a_base + buf_a_size
    end_b = buf_b_base + buf_b_size
    if not (end_a <= buf_b_base or end_b <= buf_a_base):
        raise ValueError(
            f"Ping-pong buffers overlap: A=[0x{buf_a_base:X},0x{end_a:X}), "
            f"B=[0x{buf_b_base:X},0x{end_b:X}). Adjust ofm_base or sizes."
        )

    mp.activation_peak_bytes = buf_a_size + buf_b_size
    mp.total_size = max(buf_a_base + buf_a_size, buf_b_base + buf_b_size) - min(buf_a_base, buf_b_base)

    for cfg in layer_cfgs:
        use_a_for_ifm = (cfg.layer_id % 2 == 0)
        ifm_name = getattr(cfg, "ifm_name", None)
        ofm_name = getattr(cfg, "ofm_name", None)
        if ifm_name:
            mp.ifm_offsets[ifm_name] = buf_a_base if use_a_for_ifm else buf_b_base
        if ofm_name:
            mp.ofm_offsets[ofm_name] = buf_b_base if use_a_for_ifm else buf_a_base


def _assign_per_tensor(
    layer_cfgs: list[LayerConfig],
    mp: MemoryPlan,
    ifm_base: int,
    ofm_base: int | None,
) -> None:
    """Deprecated: use allocation_mode='arena'."""
    _assign_arena(layer_cfgs, mp, ifm_base, ofm_base)


def _assign_arena(
    layer_cfgs: list[LayerConfig],
    mp: MemoryPlan,
    arena_base: int,
    ofm_base: int | None,
) -> None:
    """
    Assign a unique activation address per logical tensor name (no reuse).

    This mode is required for graphs with fan-out / skip connections and for
    mixed NPU/CPU execution where multiple tensors must coexist.
    """
    # Arena mode can place graph inputs at ifm_base and all produced tensors at ofm_base
    # (to preserve legacy absolute mapping conventions). In offset mode the caller
    # typically sets ofm_base=None, so both segments start at 0.
    produced_base = int(ofm_base) if ofm_base is not None else int(arena_base)

    # First, determine a stable size for each tensor name from LayerConfig IO.
    tensor_sizes: Dict[str, int] = {}
    for cfg in layer_cfgs:
        ifm_name = getattr(cfg, "ifm_name", None)
        ofm_name = getattr(cfg, "ofm_name", None)
        if ifm_name:
            sz = _activation_bytes_from_cfg(cfg, is_ifm=True)
            prev = tensor_sizes.get(ifm_name)
            if prev is not None and int(prev) != int(sz):
                raise ValueError(f"Arena tensor size mismatch for '{ifm_name}': prev={prev}, now={sz}")
            tensor_sizes[ifm_name] = sz
        if ofm_name:
            sz = _activation_bytes_from_cfg(cfg, is_ifm=False)
            prev = tensor_sizes.get(ofm_name)
            if prev is not None and int(prev) != int(sz):
                raise ValueError(f"Arena tensor size mismatch for '{ofm_name}': prev={prev}, now={sz}")
            tensor_sizes[ofm_name] = sz

    # Partition tensor names into:
    #  - graph inputs: appear as IFM but are never produced by an NPU layer
    #  - produced tensors: any tensor that is produced by an NPU layer output
    produced_names = {str(getattr(cfg, "ofm_name", "")) for cfg in layer_cfgs if getattr(cfg, "ofm_name", "")}
    input_names = {str(getattr(cfg, "ifm_name", "")) for cfg in layer_cfgs if getattr(cfg, "ifm_name", "")}
    graph_inputs = sorted([n for n in input_names if n and n not in produced_names])

    # Allocate graph inputs at arena_base; allocate produced tensors at produced_base.
    assigned: Dict[str, int] = {}

    cursor_in = int(arena_base)
    for name in graph_inputs:
        if name in assigned:
            continue
        cursor_in = (cursor_in + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
        assigned[name] = cursor_in
        cursor_in += int(tensor_sizes.get(name, 0))

    cursor_prod = produced_base
    for cfg in layer_cfgs:
        for name in (getattr(cfg, "ofm_name", None), getattr(cfg, "ifm_name", None)):
            if not name or name in assigned:
                continue
            cursor_prod = (cursor_prod + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
            assigned[str(name)] = cursor_prod
            cursor_prod += int(tensor_sizes.get(name, 0))

    # Basic overlap check between the two segments (best-effort).
    end_in = cursor_in
    end_prod = cursor_prod
    if produced_base != int(arena_base):
        a0, a1 = int(arena_base), end_in
        b0, b1 = produced_base, end_prod
        if not (a1 <= b0 or b1 <= a0):
            raise ValueError(
                f"Arena segments overlap: inputs=[0x{a0:X},0x{a1:X}) produced=[0x{b0:X},0x{b1:X}). "
                "Adjust ifm_base/ofm_base or use offset mode."
            )

    mp.activation_peak_bytes = (end_in - int(arena_base)) + (end_prod - produced_base)
    mp.total_size = max(end_in, end_prod) - min(int(arena_base), produced_base)

    mp.ifm_offsets = dict(assigned)
    mp.ofm_offsets = dict(assigned)


def _assign_arena_liveness(
    program: VcProgram,
    layer_cfgs: list[LayerConfig],
    mp: MemoryPlan,
    arena_base: int,
) -> None:
    """
    Assign activation addresses using a conservative liveness-based allocator.

    Motivation:
      - The legacy arena allocator assigns a unique buffer per tensor name,
        which can significantly over-estimate activation memory for deep models
        where most intermediates are not live at the same time.
      - This allocator reuses freed blocks based on last-use indices derived
        from the full IR program op order.

    Notes:
      - This allocator is intentionally conservative: it does not reuse memory
        freed on the same op index for that op's outputs (no in-place writes).
      - It only considers tensors that appear as IFM/OFM names in LayerConfig,
        which are the tensors the backend uOPs will reference.
      - Reshape/Identity/Flatten are treated as storage aliases when present in
        program.metadata["storage_alias"] (and/or IR alias ops).
    """
    from venuscore_compiler.ir.ops import VcFlatten, VcIdentity, VcReshape

    def _align_up(v: int, alignment: int = ALIGNMENT) -> int:
        return (v + alignment - 1) & ~(alignment - 1) if alignment > 0 else v

    def _collect_storage_alias() -> Dict[str, str]:
        storage_alias: Dict[str, str] = {}
        meta = getattr(program, "metadata", {}) or {}
        if isinstance(meta, dict):
            sa = meta.get("storage_alias", {})
            if isinstance(sa, dict):
                storage_alias.update({str(k): str(v) for k, v in sa.items()})
        for op in program.ops:
            if isinstance(op, (VcIdentity, VcReshape, VcFlatten)):
                if getattr(op, "inputs", None) and getattr(op, "outputs", None):
                    storage_alias[str(op.outputs[0])] = str(op.inputs[0])
        return storage_alias

    def _resolve_alias_root(name: str, storage_alias: Dict[str, str]) -> str:
        cur = str(name)
        seen = set()
        while cur in storage_alias:
            if cur in seen:
                raise ValueError(f"[memory] storage_alias cycle detected at '{cur}'")
            seen.add(cur)
            cur = str(storage_alias[cur])
        return cur

    storage_alias = _collect_storage_alias()

    # Relevant tensor names are those referenced by NPU layers (LayerConfig IO).
    names: set[str] = set()
    for cfg in layer_cfgs:
        if getattr(cfg, "ifm_name", None):
            names.add(str(cfg.ifm_name))
        if getattr(cfg, "ofm_name", None):
            names.add(str(cfg.ofm_name))

    if not names:
        mp.activation_peak_bytes = 0
        mp.total_size = 0
        mp.ifm_offsets = {}
        mp.ofm_offsets = {}
        return

    roots = {(_resolve_alias_root(n, storage_alias)) for n in names}

    # Derive stable sizes from LayerConfig IO (NCHWc4 bytes), keyed by storage root.
    root_size: Dict[str, int] = {}
    for cfg in layer_cfgs:
        ifm_name = getattr(cfg, "ifm_name", None)
        if ifm_name:
            r = _resolve_alias_root(str(ifm_name), storage_alias)
            sz = int(_activation_bytes_from_cfg(cfg, is_ifm=True))
            prev = root_size.get(r)
            if prev is not None and prev != sz:
                raise ValueError(f"[memory] Arena tensor size mismatch for root '{r}': prev={prev}, now={sz}")
            root_size[r] = sz
        ofm_name = getattr(cfg, "ofm_name", None)
        if ofm_name:
            r = _resolve_alias_root(str(ofm_name), storage_alias)
            sz = int(_activation_bytes_from_cfg(cfg, is_ifm=False))
            prev = root_size.get(r)
            if prev is not None and prev != sz:
                raise ValueError(f"[memory] Arena tensor size mismatch for root '{r}': prev={prev}, now={sz}")
            root_size[r] = sz

    # Compute last-use indices over the full IR program order, but only for the roots we care about.
    producer: Dict[str, int] = {}
    last_use: Dict[str, int] = {}
    for idx, op in enumerate(program.ops):
        for t in getattr(op, "inputs", []) or []:
            r = _resolve_alias_root(str(t), storage_alias)
            if r in roots:
                last_use[r] = max(last_use.get(r, -1), idx)
        # Only count non-alias ops as producers; alias ops do not allocate new storage.
        is_alias_op = isinstance(op, (VcIdentity, VcReshape, VcFlatten))
        if not is_alias_op:
            for t in getattr(op, "outputs", []) or []:
                r = _resolve_alias_root(str(t), storage_alias)
                if r in roots:
                    if r in producer and producer[r] != idx:
                        raise ValueError(f"[memory] Storage root '{r}' has multiple producers: {producer[r]} and {idx}")
                    producer[r] = idx

    # Roots that are produced but never consumed are treated as graph outputs: keep until the end.
    end_idx = max(0, len(program.ops) - 1)
    for r in list(roots):
        if r in producer and r not in last_use:
            last_use[r] = end_idx

    # Roots that are never used at all can be ignored.
    live_roots = {r for r in roots if r in last_use or r in producer}

    # Allocate graph inputs (used but not produced) up front for determinism.
    graph_inputs = sorted([r for r in live_roots if r in last_use and r not in producer])

    def _coalesce_free(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not blocks:
            return []
        blocks = sorted(blocks, key=lambda x: x[0])
        out: list[tuple[int, int]] = []
        cur_off, cur_sz = blocks[0]
        for off, sz in blocks[1:]:
            if cur_off + cur_sz == off:
                cur_sz += sz
            else:
                out.append((cur_off, cur_sz))
                cur_off, cur_sz = off, sz
        out.append((cur_off, cur_sz))
        return out

    free_list: list[tuple[int, int]] = []
    off_by_root: Dict[str, int] = {}
    cursor = int(arena_base)

    def _alloc_block(sz_bytes: int) -> int:
        nonlocal cursor, free_list
        need = _align_up(int(sz_bytes), ALIGNMENT)
        # First-fit in free list.
        for i, (off, sz) in enumerate(free_list):
            if sz >= need:
                alloc_off = off
                rem = sz - need
                del free_list[i]
                if rem:
                    free_list.append((off + need, rem))
                    free_list = _coalesce_free(free_list)
                return alloc_off
        cursor = _align_up(cursor, ALIGNMENT)
        alloc_off = cursor
        cursor += need
        return alloc_off

    # Allocate graph inputs.
    for r in graph_inputs:
        sz = int(root_size.get(r, 0))
        off_by_root[r] = _alloc_block(sz)

    # Pre-index produced roots per op index.
    produced_at: Dict[int, list[str]] = {}
    for r, pidx in producer.items():
        if r in live_roots:
            produced_at.setdefault(int(pidx), []).append(r)
    for pidx in produced_at:
        produced_at[pidx].sort()

    # Simulate execution order to free blocks when tensors die.
    for idx, op in enumerate(program.ops):
        for r in produced_at.get(idx, []):
            if r in off_by_root:
                continue
            sz = int(root_size.get(r, 0))
            off_by_root[r] = _alloc_block(sz)

        # Free roots whose last use is this op index.
        used_roots: set[str] = set()
        for t in getattr(op, "inputs", []) or []:
            r = _resolve_alias_root(str(t), storage_alias)
            if r in off_by_root:
                used_roots.add(r)

        for r in used_roots:
            if last_use.get(r, -1) == idx and r not in producer:
                # Graph inputs can be freed when last used.
                sz = _align_up(int(root_size.get(r, 0)), ALIGNMENT)
                free_list.append((off_by_root[r], sz))
        # Free produced roots as well when they reach last use (but keep graph outputs).
        for r in list(used_roots):
            if last_use.get(r, -1) == idx and r in producer:
                # Graph outputs are those with last_use forced to end_idx.
                if idx >= end_idx:
                    continue
                sz = _align_up(int(root_size.get(r, 0)), ALIGNMENT)
                free_list.append((off_by_root[r], sz))
        if free_list:
            free_list = _coalesce_free(free_list)

    # Map all tensor names (including aliases) to their root offset.
    assigned: Dict[str, int] = {}
    for n in sorted(names):
        r = _resolve_alias_root(n, storage_alias)
        if r not in off_by_root:
            # Tensor may be an alias-only name not referenced by NPU layers; skip.
            continue
        assigned[n] = int(off_by_root[r])

    # Peak arena size is the high-water mark of the cursor.
    mp.activation_peak_bytes = int(_align_up(cursor - int(arena_base), ALIGNMENT))
    mp.total_size = int(_align_up(cursor - int(arena_base), ALIGNMENT))
    mp.ifm_offsets = dict(assigned)
    mp.ofm_offsets = dict(assigned)


def _align_up(v: int, alignment: int) -> int:
    if alignment <= 0:
        return v
    return (v + alignment - 1) & ~(alignment - 1)


def _activation_region_end(layer_cfgs: list[LayerConfig], mp: MemoryPlan) -> int:
    """
    Return the end address (exclusive) of the activation region.

    Uses LayerConfig sizes and the chosen base addresses recorded in mp.
    """
    end = 0
    for cfg in layer_cfgs:
        ifm_name = getattr(cfg, "ifm_name", None)
        ofm_name = getattr(cfg, "ofm_name", None)
        if ifm_name and ifm_name in mp.ifm_offsets:
            base = int(mp.ifm_offsets[ifm_name])
            end = max(end, base + int(_activation_bytes_from_cfg(cfg, is_ifm=True)))
        if ofm_name and ofm_name in mp.ofm_offsets:
            base = int(mp.ofm_offsets[ofm_name])
            end = max(end, base + int(_activation_bytes_from_cfg(cfg, is_ifm=False)))
    return end


def _activation_bytes_from_cfg(cfg: object | None, is_ifm: bool) -> int:
    """
    Compute activation size from LayerConfig (NCHWc4).

    Assumes LayerConfig provides:

        - ifm_h, ifm_w, c4_in for IFM
        - ofm_h, ofm_w, c4_out for OFM

    and batch N = 1 (current VenusCore design).
    """
    if cfg is None:
        return 0

    n = 1
    h = cfg.ifm_h if is_ifm else cfg.ofm_h
    w = cfg.ifm_w if is_ifm else cfg.ofm_w
    c4 = cfg.c4_in if is_ifm else cfg.c4_out

    return n * c4 * h * w * WORD_BYTES
