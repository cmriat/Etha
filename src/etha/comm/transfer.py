"""Transfer operation types and execution."""

from enum import Enum

import torch
import torch.distributed as dist

from etha.pg_utils import get_or_create_process_group


class Transport(Enum):
    """How a chunk's bytes cross ranks (orthogonal to its produce/consume role)."""

    P2P = "p2p"
    BROADCAST = "broadcast"
    LOCAL = "local"  # same-rank copy, no wire op
    NONE = "none"  # reduce-only ("shadow"): participates in a collective but ships nothing


def _execute_p2p(
    buffer: torch.Tensor,
    is_source: bool,
    src_rank: int,
    dst_rank: int,
) -> dist.Work:
    """Execute point-to-point transfer."""
    if is_source:
        return dist.isend(buffer, dst=dst_rank)
    else:
        return dist.irecv(buffer, src=src_rank)


def _execute_broadcast(
    buffer: torch.Tensor,
    src_rank: int,
    dst_ranks: tuple[int, ...],
) -> dist.Work:
    """Execute broadcast operation."""
    group_ranks = sorted([src_rank, *dst_ranks])
    group = get_or_create_process_group(group_ranks)
    return dist.broadcast(buffer, src=src_rank, group=group, async_op=True)
