"""Communication utilities for Etha."""

import logging

import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, DTensor, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Placement

from .comm_execution import execute_naive, prepare_recv_buffers, prepare_send_buffers
from .p2p_map_lowering import map_to_ops

logger = logging.getLogger(__name__)


def gather_broadcast_communicate(
    target_mesh: DeviceMesh,
    target_specs: tuple[Placement, ...],
    local_tensor: DTensor,
    origin_tensor: torch.Tensor,
    source_world_size: int,
):
    """Performs data redistribution using the Gather-Broadcast method."""
    rank = dist.get_rank()
    gathered_tensor = None
    # 1. Gather the full tensor. After this, every rank in source_mesh has a full copy.
    if rank < source_world_size:
        gathered_tensor = local_tensor.full_tensor()

    # 2. Broadcast the full tensor from a single source (rank 0) to all other ranks.
    # Ranks outside the source_mesh need a placeholder tensor to receive the data.
    if rank >= source_world_size:
        gathered_tensor = torch.empty(origin_tensor.shape, dtype=origin_tensor.dtype, device=origin_tensor.device)

    # Rank 0 broadcasts to the default process group (all ranks).
    dist.broadcast(gathered_tensor, src=0)

    # Ensure the broadcast is complete before proceeding.
    dist.barrier()

    # 3. Distribute the now-local full tensor on target ranks.
    received_tensor = None
    if rank >= source_world_size:
        received_tensor = distribute_tensor(gathered_tensor, target_mesh, target_specs)

    return received_tensor


def get_shard_shape(device_mesh: tuple[int, ...], placements: tuple[Placement, ...], tensor_ndim: int) -> list[int]:
    """Calculate shard shape from device mesh and placements."""
    shard_shape = [1] * tensor_ndim
    for i, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_shape[placement.dim] *= device_mesh[i]
    return shard_shape


def get_shard_tensor_shape(
    origin_full_shape: torch.Size, target_device_mesh: DeviceMesh, placements: tuple[Placement, ...]
) -> torch.Size:
    target_shard_shape = list(origin_full_shape)
    for i, placement in enumerate(placements):
        if isinstance(placement, Shard):
            mesh_dim_size = target_device_mesh.mesh.shape[i]
            if target_shard_shape[placement.dim] % mesh_dim_size != 0:
                raise ValueError(
                    f"Dimension {placement.dim} of tensor with shape "
                    f"{target_shard_shape[placement.dim]} is not divisible by "
                    f"mesh dimension {i} with size {mesh_dim_size}"
                )
            target_shard_shape[placement.dim] //= mesh_dim_size
    return torch.Size(target_shard_shape)


def assemble_target_tensor(
    rank: int,
    reverse_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    received_data: dict[tuple[int, tuple], torch.Tensor],
    recv_buffers: dict[tuple[tuple, int, tuple], torch.Tensor],
    target_num_slicers: list[int],
    final_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Assemble final tensor from received chunks.

    Returns:
        Final assembled tensor if rank is a receiver, None otherwise.
    """
    if rank not in reverse_map:
        return None

    for target_idx, src_list in reverse_map[rank].items():
        for src_rank, sidx in src_list:
            if src_rank == rank:
                if (src_rank, sidx) not in received_data:
                    continue
                buf = received_data[(src_rank, sidx)]
            else:
                buf_key = (target_idx, src_rank, sidx)
                buf = recv_buffers[buf_key]

            slice_ranges = []
            for dim, coord in enumerate(target_idx):
                if target_num_slicers[dim] > 1:
                    slice_size = final_tensor.shape[dim] // target_num_slicers[dim]
                    start = coord * slice_size
                    end = start + slice_size
                    slice_ranges.append(slice(start, end))
                else:
                    slice_ranges.append(slice(None))

            final_tensor[tuple(slice_ranges)].copy_(buf)

            if src_rank != rank:
                del recv_buffers[buf_key]


def p2p_communicate(
    source_local_tensor: torch.Tensor | None,
    target_local_tensor: torch.Tensor | None,
    forward_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    reverse_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    source_num_slicers: list[int],
    target_num_slicers: list[int],
) -> None:
    """Execute point-to-point communication using IR-based architecture.

    This function implements the three-tier architecture:
    1. Lowering: map_to_ops() converts topology to IR chunks
    2. Preparation: prepare_*_buffers() allocate/slice buffers
    3. Execution: execute_naive() performs actual communication

    IMPORTANT: This function modifies target_local_tensor IN-PLACE.

    Args:
        source_local_tensor: Local tensor to send from (can be None for receiver-only)
        target_local_tensor: Local tensor to receive into (modified in-place, can be None for sender-only)
        forward_map: Topology map for sending
        reverse_map: Topology map for receiving
        source_num_slicers: Partitioning of source tensor
        target_num_slicers: Partitioning of target tensor

    Returns:
        None (result is written to target_local_tensor in-place)
    """
    rank = dist.get_rank()

    # Determine role and get tensor properties
    is_sender = rank in forward_map
    is_receiver = rank in reverse_map

    if not is_sender and not is_receiver:
        return None

    # Infer target_tensor_shape, device, dtype from available tensor
    if target_local_tensor is not None:
        target_tensor_shape = tuple(target_local_tensor.shape)
        device = target_local_tensor.device
        dtype = target_local_tensor.dtype
    elif source_local_tensor is not None:
        # Sender-only: use source tensor properties as placeholder
        target_tensor_shape = tuple(source_local_tensor.shape)
        device = source_local_tensor.device
        dtype = source_local_tensor.dtype
    else:
        raise ValueError("Both source_local_tensor and target_local_tensor are None")

    # === Phase 1: Lowering (planning) ===
    source_chunks, target_chunks = map_to_ops(
        forward_map=forward_map,
        reverse_map=reverse_map,
        source_num_slicers=source_num_slicers,
        target_num_slicers=target_num_slicers,
        target_tensor_shape=target_tensor_shape,
        rank=rank,
    )

    # If no chunks to process, return None
    if not source_chunks and not target_chunks:
        return None

    # === Phase 2: Preparation (buffer allocation) ===
    if source_chunks:
        prepare_send_buffers(
            chunks=source_chunks,
            local_tensor=source_local_tensor,
            source_num_slicers=source_num_slicers,
        )

    if target_chunks:
        prepare_recv_buffers(
            chunks=target_chunks,
            source_local_tensor=source_local_tensor,  # For self-copy
            source_num_slicers=source_num_slicers,
            device=device,
            dtype=dtype,
        )

    execute_naive(
        source_chunks=source_chunks,
        target_chunks=target_chunks,
        target_tensor=target_local_tensor,
        target_num_slicers=target_num_slicers,
    )
