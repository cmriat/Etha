"""Communication utilities for Etha."""

import logging

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Placement

from .get_chunk_ir import map_to_chunk_ir
from .comm_execution import execute_naive

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
    if source_local_tensor is None and target_local_tensor is None:
        raise ValueError("Both source_local_tensor and target_local_tensor are None")

    rank = dist.get_rank()

    source_tensor_shape = None
    target_tensor_shape = None
    # Only extract shape if tensor is non-empty (has data)
    # Empty tensors (shape contains 0) indicate rank not in mesh
    if target_local_tensor is not None and 0 not in target_local_tensor.shape:
        target_tensor_shape = tuple(target_local_tensor.shape)
    if source_local_tensor is not None and 0 not in source_local_tensor.shape:
        source_tensor_shape = tuple(source_local_tensor.shape)

    # === Phase 1: Lowering (planning) ===
    source_chunks, target_chunks = map_to_chunk_ir(
        forward_map=forward_map,
        reverse_map=reverse_map,
        source_num_slicers=source_num_slicers,
        target_num_slicers=target_num_slicers,
        source_tensor_shape=source_tensor_shape,
        target_tensor_shape=target_tensor_shape,
        rank=rank,
    )

    # If no chunks to process, return None
    if not source_chunks and not target_chunks:
        return None

    # === Phase 2: Execution (prepare + send + recv + assemble) ===
    execute_naive(
        source_chunks=source_chunks,
        target_chunks=target_chunks,
        source_local_tensor=source_local_tensor,
        target_local_tensor=target_local_tensor,
    )
