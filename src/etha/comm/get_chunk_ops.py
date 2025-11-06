"""Convert transfer IR to chunk IR."""

import torch

from etha.comm.ir import SourceChunk, TargetChunk, logger

from .ir import SourceChunk, TargetChunk, TransferType
from .utils import (
    get_slicer_tuples,
    get_slice_from_multi_index,
    get_or_create_process_group,
)


def calculate_chunk_shape(
    num_slicers: list[int],
    tensor_shape: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Calculate chunk shape from num_slicers and tensor shape.

    Args:
        num_slicers: Number of slices per dimension
        tensor_shape: Full tensor shape (if known)

    Returns:
        Shape of the chunk, or empty tuple if tensor_shape is None
    """
    if tensor_shape is None:
        return ()

    # Extend num_slicers to match tensor dimensions
    num_slicers = num_slicers + [1] * (len(tensor_shape) - len(num_slicers))

    chunk_shape = tuple(tensor_shape[dim] // num_slicers[dim] for dim in range(len(tensor_shape)))

    return chunk_shape


def bind_tensors_to_chunks(
    source_chunks: list[SourceChunk],
    target_chunks: list[TargetChunk],
    source_tensor: torch.Tensor | None = None,
    target_tensor: torch.Tensor | None = None,
) -> None:
    """Bind tensor references to chunks (in-place operation).

    This function transitions chunks from planning phase (tensor-agnostic) to
    execution phase (tensor-bound). After binding, chunks are self-contained
    and can be executed without passing tensors separately.

    Args:
        source_chunks: List of source chunks to bind (modified in-place)
        target_chunks: List of target chunks to bind (modified in-place)
        source_tensor: Tensor to bind to source chunks (None for receiver-only ranks)
        target_tensor: Tensor to bind to target chunks (None for sender-only ranks)

    Note:
        This is an in-place operation. After calling this function, the chunks'
        .tensor field will reference the provided tensors.
    """
    if source_tensor is not None:
        for chunk in source_chunks:
            chunk.tensor = source_tensor
            logger.debug("Bound source tensor to chunk")

    if target_tensor is not None:
        for chunk in target_chunks:
            chunk.tensor = target_tensor
            logger.debug("Bound target tensor to chunk")


def map_to_chunk_ops(
    m2m_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    rank: int,
    source_num_slicers: list[int],
    target_num_slicers: list[int],
    source_tensor: torch.Tensor | None = None,
    target_tensor: torch.Tensor | None = None,
) -> list[SourceChunk | TargetChunk]:
    """Convert M2MMap directly to execution-ready chunks.

    Eliminates Transfer abstraction by directly generating chunks from M2MMap.
    Returns a unified list of chunks (both source and target operations).

    Args:
        m2m_map: Communication topology {src_rank: {src_idx: [(dst_rank, dst_idx), ...]}}
        rank: Current process rank
        source_num_slicers: How source tensor is partitioned
        target_num_slicers: How target tensor is partitioned
        source_tensor: Tensor to bind for source chunks
        target_tensor: Tensor to bind for target chunks

    Returns:
        Unified list of SourceChunk and TargetChunk objects
    """
    if not m2m_map:
        return []

    # Get tensor shapes from provided tensors
    source_tensor_shape = source_tensor.shape if source_tensor is not None else None
    target_tensor_shape = target_tensor.shape if target_tensor is not None else None

    # Pre-calculate slicer tuples for slice indexing
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

    # Pre-create broadcast process groups
    broadcast_groups = set()
    for src_rank, src_map in m2m_map.items():
        for _, dst_list in src_map.items():
            if len(dst_list) > 1:
                dst_ranks = sorted({r for r, _ in dst_list})
                group_ranks = tuple([src_rank] + dst_ranks)
                broadcast_groups.add(group_ranks)

    for group_ranks in sorted(broadcast_groups):
        get_or_create_process_group(list(group_ranks))

    chunks: list[SourceChunk | TargetChunk] = []

    # Generate chunks from m2m_map
    for src_rank, src_map in m2m_map.items():
        for src_idx, dst_list in src_map.items():
            # Determine transfer type
            if len(dst_list) > 1:
                transfer_type = TransferType.BROADCAST
            else:
                transfer_type = TransferType.P2P

            dst_ranks = sorted({r for r, _ in dst_list})

            # === Generate SourceChunk if this rank is the sender ===
            if src_rank == rank:
                group_key = (rank, tuple(dst_ranks)) if transfer_type == TransferType.BROADCAST else None

                slice_tuples = ()
                if source_slicer_tuples is not None:
                    slice_tuples = get_slice_from_multi_index(
                        src_idx, source_num_slicers_extended, source_slicer_tuples
                    )

                source_chunk = SourceChunk(
                    chunk_shape=calculate_chunk_shape(source_num_slicers, source_tensor_shape),
                    transfer_type=transfer_type,
                    src_rank=rank,
                    src_idx=src_idx,
                    dst_ranks=dst_ranks,
                    group_key=group_key,
                    slice_tuples=slice_tuples,
                    tensor=source_tensor,
                )

                chunks.append(source_chunk)

            # === Generate TargetChunks for each destination ===
            for dst_rank, dst_idx in dst_list:
                if dst_rank == rank:
                    if src_rank == rank:
                        # Self-copy case
                        actual_transfer_type = TransferType.SELF_COPY
                        group_key = None
                    else:
                        actual_transfer_type = transfer_type
                        if transfer_type == TransferType.BROADCAST:
                            group_key = (src_rank, tuple(dst_ranks))
                        else:
                            group_key = None

                    # Pre-calculate slice_tuples
                    slice_tuples = ()
                    if target_slicer_tuples is not None:
                        slice_tuples = get_slice_from_multi_index(
                            dst_idx, target_num_slicers_extended, target_slicer_tuples
                        )

                    # Pre-calculate src_slice_tuples for self_copy
                    src_slice_tuples = ()
                    if actual_transfer_type == TransferType.SELF_COPY and source_slicer_tuples is not None:
                        src_slice_tuples = get_slice_from_multi_index(
                            src_idx, source_num_slicers_extended, source_slicer_tuples
                        )

                    target_chunk = TargetChunk(
                        chunk_shape=calculate_chunk_shape(target_num_slicers, target_tensor_shape),
                        transfer_type=actual_transfer_type,
                        dst_rank=rank,
                        dst_idx=dst_idx,
                        src_rank=src_rank,
                        src_idx=src_idx,
                        group_key=group_key,
                        slice_tuples=slice_tuples,
                        src_slice_tuples=src_slice_tuples,
                        tensor=target_tensor,
                    )

                    chunks.append(target_chunk)

    return chunks
