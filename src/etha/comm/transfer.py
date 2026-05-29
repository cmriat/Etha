"""Transfer operation types and execution."""

from enum import Enum
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .utils import get_or_create_process_group


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


@dataclass(slots=True, kw_only=True)
class Transferable:
    """Base class for transferable objects (chunks and buckets).

    Role and transport are orthogonal:
    - ``is_source``: reads a local tensor into ``buffer`` (source side / self-copy).
    - ``is_target``: writes ``buffer`` back into a local target tensor (recv / self-copy).
    - ``transport``: how bytes cross ranks. ``LOCAL``/``NONE`` never hit the wire.
    """

    transport: Transport
    is_source: bool
    is_target: bool
    src_rank: int
    dst_ranks: tuple[int, ...]
    buffer: torch.Tensor | None = None
    work: dist.Work | None = None

    def execute(self) -> dist.Work | None:
        """Execute transfer operation.

        Returns:
            Work handle for async transports, None for LOCAL / NONE.
        """
        match self.transport:
            case Transport.LOCAL | Transport.NONE:
                return None
            case Transport.P2P:
                return _execute_p2p(self.buffer, self.is_source, self.src_rank, self.dst_ranks[0])
            case Transport.BROADCAST:
                return _execute_broadcast(self.buffer, self.src_rank, self.dst_ranks)
