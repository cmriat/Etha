"""Convert transfer IR to chunk IR."""

import torch

from etha.comm.ir import SourceChunk, TargetChunk, logger

from .ir import Transfer, SourceChunk, TargetChunk, TransferType
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


def build_broadcast_groups(transfers: list[Transfer]) -> None:
    """Pre-create process groups for all broadcast operations.

    Creates groups in deterministic order across all ranks to avoid deadlocks.

    Args:
        transfers: List of Transfer objects
    """
    broadcast_groups = set()
    for transfer in transfers:
        if transfer.transfer_type == TransferType.BROADCAST:
            dst_ranks = sorted({r for r, _ in transfer.dst_list})
            group_ranks = tuple([transfer.src_rank] + dst_ranks)
            broadcast_groups.add(group_ranks)

    for group_ranks in sorted(broadcast_groups):
        get_or_create_process_group(list(group_ranks))


def transfers_to_chunks(
    transfers: list[Transfer],
    rank: int,
    source_tensor_shape: tuple[int, ...] | None,
    target_tensor_shape: tuple[int, ...] | None,
) -> tuple[list[SourceChunk], list[TargetChunk]]:
    """Convert Transfer IR to execution-ready Chunks for this rank.

    Args:
        transfers: List of all Transfer objects (global)
        rank: Current process rank
        source_tensor_shape: Shape of source tensor (if known)
        target_tensor_shape: Shape of target tensor (if known)

    Returns:
        (source_chunks, target_chunks) for this rank
    """
    # Get num_slicers from first Transfer (all transfers have same values)
    if not transfers:
        return [], []

    source_num_slicers = transfers[0].source_num_slicers
    target_num_slicers = transfers[0].target_num_slicers

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

    # Pre-create broadcast process groups before generating chunks
    # This ensures all ranks create groups in the same order
    build_broadcast_groups(transfers)

    source_chunks: list[SourceChunk] = []
    target_chunks: list[TargetChunk] = []

    # === Generate SourceChunks from transfers where src_rank == rank ===
    for transfer in transfers:
        if transfer.src_rank == rank:
            dst_ranks = sorted({r for r, _ in transfer.dst_list})
            group_key = (rank, tuple(dst_ranks)) if transfer.transfer_type == TransferType.BROADCAST else None

            slice_tuples = ()
            if source_slicer_tuples is not None:
                slice_tuples = get_slice_from_multi_index(
                    transfer.src_idx, source_num_slicers_extended, source_slicer_tuples
                )

            source_chunk = SourceChunk(
                chunk_shape=calculate_chunk_shape(source_num_slicers, source_tensor_shape),
                transfer_type=transfer.transfer_type,
                src_rank=rank,
                src_idx=transfer.src_idx,
                dst_ranks=dst_ranks,
                group_key=group_key,
                slice_tuples=slice_tuples,
                stage_id=transfer.stage_id,
            )

            source_chunks.append(source_chunk)

    # === Generate TargetChunks from transfers===
    for transfer in transfers:
        for dst_rank, dst_idx in transfer.dst_list:
            if dst_rank == rank:
                if transfer.src_rank == rank:
                    transfer_type = TransferType.SELF_COPY
                    group_key = None
                else:
                    transfer_type = transfer.transfer_type
                    if transfer_type == TransferType.BROADCAST:
                        dst_ranks = sorted({r for r, _ in transfer.dst_list})
                        group_key = (transfer.src_rank, tuple(dst_ranks))
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
                if transfer_type == TransferType.SELF_COPY and source_slicer_tuples is not None:
                    src_slice_tuples = get_slice_from_multi_index(
                        transfer.src_idx, source_num_slicers_extended, source_slicer_tuples
                    )

                target_chunk = TargetChunk(
                    chunk_shape=calculate_chunk_shape(target_num_slicers, target_tensor_shape),
                    transfer_type=transfer_type,
                    dst_rank=rank,
                    dst_idx=dst_idx,
                    src_rank=transfer.src_rank,
                    src_idx=transfer.src_idx,
                    group_key=group_key,
                    slice_tuples=slice_tuples,
                    src_slice_tuples=src_slice_tuples,
                    stage_id=transfer.stage_id,
                )

                target_chunks.append(target_chunk)

    return source_chunks, target_chunks


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
