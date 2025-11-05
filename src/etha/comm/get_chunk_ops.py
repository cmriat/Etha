"""Communication Lowering - Convert topology maps to IR chunks."""

import torch
from torch.distributed import ProcessGroup
from torch.distributed.tensor import Placement, DeviceMesh

from etha.comm.chunk_ops import SourceChunk, TargetChunk, logger

from .utils import (
    get_slicer_tuples,
    get_slice_from_multi_index,
    get_or_create_process_group,
)
from .chunk_ops import Transfer, SourceChunk, TargetChunk, TransferType


def get_m2m_transfers(
    source_mesh: DeviceMesh,
    source_placements: tuple[Placement, ...],
    target_mesh: DeviceMesh,
    target_placements: tuple[Placement, ...],
    group: ProcessGroup,
    device: str = "cpu",
) -> list[Transfer]:
    """Generate Transfer IR for mesh-to-mesh communication.

    This is the primary user-facing function for generating a complete communication
    plan. Each Transfer contains all necessary metadata including partitioning info.

    Args:
        source_mesh: Source device mesh
        source_placements: Source tensor placements
        target_mesh: Target device mesh
        target_placements: Target tensor placements
        group: Process group for communication
        device: Device for execution ("cpu" or "cuda")

    Returns:
        List of Transfer objects with complete metadata
    """
    from .get_m2m_map import get_m2m_map

    # Generate maps
    forward_map, source_num_slicers, target_num_slicers = get_m2m_map(
        source_mesh=source_mesh,
        source_placements=source_placements,
        target_mesh=target_mesh,
        target_placements=target_placements,
        group=group,
        device=device,
    )

    # Convert to Transfer IR
    return map_to_transfers(forward_map, source_num_slicers, target_num_slicers)


def map_to_transfers(
    forward_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    source_num_slicers: list[int],
    target_num_slicers: list[int],
) -> list[Transfer]:
    """Convert forward_map to Transfer IR with complete metadata.

    Args:
        forward_map: src_rank -> {src_idx: [(dst_rank, dst_idx), ...]}
        source_num_slicers: Partitioning of source tensor
        target_num_slicers: Partitioning of target tensor

    Returns:
        List of Transfer objects with complete metadata (excluding self-copy)
    """
    transfers = []
    for src_rank in sorted(forward_map.keys()):
        for src_idx in sorted(forward_map[src_rank].keys()):
            dst_list = forward_map[src_rank][src_idx]

            other_dsts = [(r, idx) for r, idx in dst_list if r != src_rank]

            if other_dsts:
                # Determine transfer type based on number of destinations
                transfer_type = TransferType.BROADCAST if len(other_dsts) > 1 else TransferType.P2P

                transfers.append(
                    Transfer(
                        src_rank=src_rank,
                        src_idx=src_idx,
                        dst_list=other_dsts,
                        transfer_type=transfer_type,
                        source_num_slicers=source_num_slicers,
                        target_num_slicers=target_num_slicers,
                    )
                )

    return transfers


def assign_pipeline_stages(
    transfers: list[Transfer],
    chunks_per_stage: int = 4,
) -> None:
    """Assign pipeline stage IDs to transfers for pipelined execution.

    Args:
        transfers: List of Transfer objects to assign stages to
        chunks_per_stage: Number of operations per pipeline stage
    """
    sorted_transfers = sorted(transfers, key=lambda t: (t.src_rank, t.src_idx))

    for i, transfer in enumerate(sorted_transfers):
        transfer.stage_id = i // chunks_per_stage


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
    chunk_id = 0

    # === Generate SourceChunks from transfers where src_rank == rank ===
    for transfer in transfers:
        if transfer.src_rank == rank:
            # Determine group_key for broadcast
            group_key = None
            if transfer.transfer_type == TransferType.BROADCAST:
                dst_ranks = sorted({r for r, _ in transfer.dst_list})
                group_key = (rank, tuple(dst_ranks))

            # Pre-calculate slice_tuples
            slice_tuples = ()
            if source_slicer_tuples is not None:
                slice_tuples = get_slice_from_multi_index(
                    transfer.src_idx, source_num_slicers_extended, source_slicer_tuples
                )

            source_chunk = SourceChunk(
                chunk_id=chunk_id,
                chunk_shape=calculate_chunk_shape(source_num_slicers, source_tensor_shape),
                transfer_type=transfer.transfer_type,
                src_rank=rank,
                src_idx=transfer.src_idx,
                dst_ranks=sorted({r for r, _ in transfer.dst_list}),
                group_key=group_key,
                slice_tuples=slice_tuples,
                stage_id=transfer.stage_id,
            )

            source_chunks.append(source_chunk)
            chunk_id += 1

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
                    chunk_id=chunk_id,
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
                chunk_id += 1

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
            logger.debug(f"Bound source tensor to chunk {chunk.chunk_id}")

    if target_tensor is not None:
        for chunk in target_chunks:
            chunk.tensor = target_tensor
            logger.debug(f"Bound target tensor to chunk {chunk.chunk_id}")
