"""Transfer operation types and execution."""

from enum import Enum
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .utils import get_or_create_process_group


class TransferType(Enum):
    """Transfer operation types."""

    SELF_COPY = "self_copy"  # Local copy within same rank
    P2P = "p2p"  # Point-to-point transfer between two ranks
    BROADCAST = "broadcast"  # One-to-many transfer


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


@dataclass(slots=True, kw_only=True)
class Transferable:
    """Base class for transferable objects (chunks and buckets)."""

    transfer_type: TransferType
    is_source: bool
    src_rank: int
    dst_ranks: tuple[int, ...]
    buffer: torch.Tensor | None = None
    work: dist.Work | None = None

    def execute(self) -> dist.Work | None:
        """Execute transfer operation.

        Returns:
            Work handle for async operations, None for SELF_COPY.
        """
        match self.transfer_type:
            case TransferType.SELF_COPY:
                return None
            case TransferType.P2P:
                return _execute_p2p(self.buffer, self.is_source, self.src_rank, self.dst_ranks[0])
            case TransferType.BROADCAST:
                return _execute_broadcast(self.buffer, self.src_rank, self.dst_ranks)
