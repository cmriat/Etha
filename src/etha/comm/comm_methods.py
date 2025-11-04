"""Communication utilities for Etha."""

import logging

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Placement

from .chunk_ir import SourceChunk, TargetChunk
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


def m2m_communicate(
    source_chunks: list[SourceChunk],
    target_chunks: list[TargetChunk],
    source_local_tensor: torch.Tensor | None,
    target_local_tensor: torch.Tensor | None,
) -> None:
    """Execute mesh-to-mesh communication using pre-compiled chunk IR.

    This is the execution phase - performs actual data transfer based on IR.

    IMPORTANT: This function modifies target_local_tensor IN-PLACE.

    Args:
        source_chunks: Chunks this rank needs to SEND (from map_to_chunk_ir)
        target_chunks: Chunks this rank needs to RECEIVE (from map_to_chunk_ir)
        source_local_tensor: Local tensor to send from (can be None for receiver-only)
        target_local_tensor: Local tensor to receive into (modified in-place, can be None for sender-only)

    Returns:
        None (result is written to target_local_tensor in-place)
    """
    # Execute transfer
    execute_naive(
        source_chunks=source_chunks,
        target_chunks=target_chunks,
        source_local_tensor=source_local_tensor,
        target_local_tensor=target_local_tensor,
    )
