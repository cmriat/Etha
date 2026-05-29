"""Communication utilities for Etha."""

import logging
from collections import deque, defaultdict

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Placement

from .ir import Bucket

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


def bucket_comm(
    buckets: list[Bucket],
    max_in_flight: int = 2,
) -> None:
    """Run the bucket pipeline: prepare -> launch -> complete -> finalize.

    Per channel key, at most ``max_in_flight`` buckets are prepared/in-flight at
    once, so buffer assembly overlaps with in-flight collectives.
    """
    rank = dist.get_rank()

    channels: defaultdict[tuple, dict[str, deque]] = defaultdict(
        lambda: {
            "candidate": deque(),
            "prepared": deque(),
            "in_flight": deque(),
        }
    )
    for bucket in buckets:
        channels[bucket.key]["candidate"].append(bucket)
    logger.debug(f"[rank={rank}] Bucket execution: Key states: {channels}")

    def _has_work() -> bool:
        for channel in channels.values():
            if channel["candidate"] or channel["prepared"] or channel["in_flight"]:
                return True
        return False

    while _has_work():
        made_progress = True
        while made_progress:
            made_progress = False
            for channel in channels.values():
                candidate: deque = channel["candidate"]
                prepared: deque = channel["prepared"]
                in_flight: deque = channel["in_flight"]

                if candidate and len(prepared) + len(in_flight) < max_in_flight:
                    bucket = candidate.popleft()
                    logger.debug(f"[rank={rank}] Bucket execution: Preparing bucket {bucket}")
                    bucket.prepare()
                    prepared.append(bucket)
                    made_progress = True

        made_progress = True
        while made_progress:
            made_progress = False
            for channel in channels.values():
                prepared: deque = channel["prepared"]
                in_flight: deque = channel["in_flight"]

                if prepared and prepared[0].launch():
                    logger.debug(f"[rank={rank}] Bucket execution: Launching bucket {prepared[0]}")
                    in_flight.append(prepared.popleft())
                    made_progress = True

        made_progress = True
        while made_progress:
            made_progress = False
            for channel in channels.values():
                in_flight: deque = channel["in_flight"]
                if in_flight and in_flight[0].is_complete():
                    logger.debug(f"[rank={rank}] Bucket execution: Finalizing bucket {in_flight[0]}")
                    in_flight.popleft().finalize()
                    made_progress = True

    torch.cuda.synchronize()
