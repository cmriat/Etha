"""Build chunk descriptors from routes."""

from collections.abc import Iterable

import torch
import torch.distributed as dist

from etha.pg_utils import get_or_create_process_group

from .ir import Chunk, M2MMap, Transport
from .utils import get_slicer_tuples, get_slice_from_multi_index


def _broadcast_group_ranks(m2m: M2MMap) -> set[tuple[int, ...]]:
    """Canonical broadcast-group rank tuples for one direction's ``M2MMap``.

    Each BROADCAST route contributes its complete, sorted membership tuple —
    the same key the process-group cache is indexed by. Sorting the source
    together with the destinations also deduplicates the same process group
    when opposite directions use different broadcast roots.
    """
    groups: set[tuple[int, ...]] = set()
    for route in m2m.routes or []:
        if route.kind == Transport.BROADCAST:
            groups.add(tuple(sorted({route.src.rank} | {dst.rank for dst in route.dsts})))
    return groups


def prewarm_broadcast_groups(m2m_maps: Iterable[M2MMap | None]) -> None:
    """Create every broadcast group across the given maps in one canonical pass.

    ``dist.new_group`` is collective on WORLD, so every rank must issue the same
    sequence of calls. ``m2m_to_chunks`` creates a direction's groups on first
    touch, and the two sides of a pair materialize chunks in opposite direction
    orders (each walks local-send before local-recv), so per-direction
    first-touch creation interleaves the calls differently on the two sides once
    both directions broadcast — silently cross-wiring the communicators. Route
    tables are identical on both sides (merged via ``all_gather_object`` in
    ``get_m2m_map``), so this union-then-sort pass runs identically everywhere
    and turns the later per-direction creations into cache hits.
    """
    groups: set[tuple[int, ...]] = set()
    for m2m in m2m_maps:
        if m2m is not None:
            groups |= _broadcast_group_ranks(m2m)
    for group_ranks in sorted(groups):
        get_or_create_process_group(list(group_ranks))


def calculate_chunk_shape(
    num_slicers: list[int],
    tensor_shape: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if tensor_shape is None:
        return ()
    chunk_shape = tuple(tensor_shape[dim] // num_slicers[dim] for dim in range(len(tensor_shape)))
    return chunk_shape


def m2m_to_chunks(
    m2m: M2MMap,
    rank: int,
    source_tensor: torch.Tensor | None = None,
    target_tensor: torch.Tensor | None = None,
    transfer_dtype: torch.dtype | None = None,
    source_partial_groups: list[tuple[dist.ProcessGroup, str]] | None = None,
) -> list[Chunk]:
    """Materialize an ``M2MMap`` (topology) onto local tensors into chunks."""
    routes = m2m.routes
    if not routes:
        return []
    source_num_slicers = m2m.source_num_slicers
    target_num_slicers = m2m.target_num_slicers
    source_tensor_shape = source_tensor.shape if source_tensor is not None else None
    target_tensor_shape = target_tensor.shape if target_tensor is not None else None
    source_slicer_tuples = None
    source_num_slicers_extended = None
    if source_tensor_shape is not None:
        source_num_slicers_extended = (source_num_slicers + [1] * len(source_tensor_shape))[: len(source_tensor_shape)]
        source_slicer_tuples = get_slicer_tuples(source_tensor_shape, source_num_slicers_extended)
    target_slicer_tuples = None
    target_num_slicers_extended = None
    if target_tensor_shape is not None:
        target_num_slicers_extended = (target_num_slicers + [1] * len(target_tensor_shape))[: len(target_tensor_shape)]
        target_slicer_tuples = get_slicer_tuples(target_tensor_shape, target_num_slicers_extended)
    for group_ranks in sorted(_broadcast_group_ranks(m2m)):
        get_or_create_process_group(list(group_ranks))

    chunks: list[Chunk] = []
    for route in routes:
        src_rank = route.src.rank
        src_idx = route.src.cell
        transport = route.kind
        dst_ranks: tuple[int, ...] = tuple(sorted({d.rank for d in route.dsts}))
        src_slice_tuples: tuple[slice, ...] = ()
        if src_rank == rank:
            if source_slicer_tuples is not None and source_num_slicers_extended is not None:
                src_slice_tuples = get_slice_from_multi_index(
                    src_idx, source_num_slicers_extended, source_slicer_tuples
                )

            chunks.append(
                Chunk(
                    chunk_shape=calculate_chunk_shape(source_num_slicers_extended, source_tensor_shape),
                    transport=transport,
                    is_source=True,
                    is_target=False,
                    src_rank=rank,
                    src_idx=src_idx,
                    dst_ranks=dst_ranks,
                    src_slice=src_slice_tuples,
                    tensor=source_tensor,
                    transfer_dtype=transfer_dtype,
                    source_partial_groups=source_partial_groups,
                )
            )
        for dst in route.dsts:
            dst_rank = dst.rank
            dst_idx = dst.cell
            if dst_rank != rank:
                continue
            dst_slice_tuples: tuple[slice, ...] = ()
            if target_slicer_tuples is not None and target_num_slicers_extended is not None:
                dst_slice_tuples = get_slice_from_multi_index(
                    dst_idx, target_num_slicers_extended, target_slicer_tuples
                )
            if src_rank == rank:
                # dst landed on the source rank: read source, write target locally.
                chunks.append(
                    Chunk(
                        chunk_shape=calculate_chunk_shape(target_num_slicers_extended, target_tensor_shape),
                        transport=Transport.LOCAL,
                        is_source=True,
                        is_target=True,
                        src_rank=src_rank,
                        src_idx=src_idx,
                        dst_ranks=dst_ranks,
                        dst_idx=dst_idx,
                        src_slice=src_slice_tuples,
                        dst_slice=dst_slice_tuples,
                        tensor=target_tensor,
                    )
                )
            else:
                chunks.append(
                    Chunk(
                        chunk_shape=calculate_chunk_shape(target_num_slicers_extended, target_tensor_shape),
                        transport=transport,
                        is_source=False,
                        is_target=True,
                        src_rank=src_rank,
                        src_idx=src_idx,
                        dst_ranks=dst_ranks,
                        dst_idx=dst_idx,
                        dst_slice=dst_slice_tuples,
                        tensor=target_tensor,
                        transfer_dtype=transfer_dtype,
                    )
                )
    return chunks
