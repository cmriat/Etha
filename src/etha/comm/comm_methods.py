"""Communication utilities for Etha."""

import logging

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Placement

from .ir import Chunk, Bucket
from .execution import execute_chunk_simple, execute_bucket_pipeline

logger = logging.getLogger(__name__)


def gather_broadcast_comm(
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
    with torch.profiler.record_function("gather_broadcast::gather_phase"):
        if rank < source_world_size:
            gathered_tensor = local_tensor.full_tensor()

    # 2. Broadcast the full tensor from a single source (rank 0) to all other ranks.
    # Ranks outside the source_mesh need a placeholder tensor to receive the data.
    with torch.profiler.record_function("gather_broadcast::broadcast_phase"):
        if rank >= source_world_size:
            gathered_tensor = torch.empty(origin_tensor.shape, dtype=origin_tensor.dtype, device=origin_tensor.device)

        # Rank 0 broadcasts to the default process group (all ranks).
        dist.broadcast(gathered_tensor, src=0)

        # Ensure the broadcast is complete before proceeding.
        dist.barrier()

    # 3. Distribute the now-local full tensor on target ranks.
    with torch.profiler.record_function("gather_broadcast::distribute_phase"):
        received_tensor = None
        if rank >= source_world_size:
            received_tensor = distribute_tensor(gathered_tensor, target_mesh, target_specs)

    return received_tensor


def chunk_comm(
    chunks: list[Chunk],
) -> None:
    """Execute chunked communication."""
    execute_chunk_simple(chunks=chunks)


def bucket_comm(
    buckets: list[Bucket],
) -> None:
    """Execute bucketized communication."""
    execute_bucket_pipeline(buckets=buckets)
