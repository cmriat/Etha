"""Build chunk descriptors from routes."""

import torch
import torch.distributed as dist

from etha.pg_utils import get_or_create_process_group

from .ir import Chunk, M2MMap, Transport
from .utils import get_slicer_tuples, get_slice_from_multi_index


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
    # Partial transfers must not coalesce; set on both ends (this is M2MMap-level, so
    # source and target ranks agree, keeping send/recv bucketing symmetric).
    no_coalesce = bool(m2m.source_partial_reductions)
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
    broadcast_groups = set()

    for route in routes:
        if route.kind == Transport.BROADCAST:
            group_ranks = (route.src.rank,) + tuple(sorted({d.rank for d in route.dsts}))
            broadcast_groups.add(group_ranks)
    for group_ranks in sorted(broadcast_groups):
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
                    no_coalesce=no_coalesce,
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
                        no_coalesce=no_coalesce,
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
                        no_coalesce=no_coalesce,
                    )
                )
    return chunks
