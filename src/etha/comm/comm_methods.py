"""Communication utilities for Etha."""

import logging

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Placement

from .ir import SourceChunk, TargetChunk
from .comm_execution import execute_naive, execute_pipelined

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
    pipelined: bool = False,
) -> None:
    """Execute mesh-to-mesh communication using pre-compiled chunk IR.

    This function performs the execution phase of mesh-to-mesh communication.
    Chunks must have tensor references bound before calling this function.

    IMPORTANT: Call bind_tensors_to_chunks() before this function to attach
    tensor references to chunks.

    Args:
        source_chunks: Chunks to send (must have .tensor bound)
        target_chunks: Chunks to receive (must have .tensor bound)

    Returns:
        None (result is written to target tensor in-place)
    """
    if pipelined:
        execute_pipelined(source_chunks=source_chunks, target_chunks=target_chunks)
    else:
        execute_naive(source_chunks=source_chunks, target_chunks=target_chunks)
