"""Transfer operation execution functions.

Separates communication execution from data management.
"""

import torch
import torch.distributed as dist

from .ir import TransferType
from .utils import get_or_create_process_group


def execute_p2p_send(buffer: torch.Tensor, dst_rank: int) -> dist.Work:
    """Execute point-to-point send.

    Args:
        buffer: Data buffer to send
        dst_rank: Destination rank

    Returns:
        Work handle for async operation
    """
    return dist.isend(buffer, dst=dst_rank)


def execute_p2p_recv(buffer: torch.Tensor, src_rank: int) -> dist.Work:
    """Execute point-to-point receive.

    Args:
        buffer: Buffer to receive data into
        src_rank: Source rank

    Returns:
        Work handle for async operation
    """
    return dist.irecv(buffer, src=src_rank)


def execute_broadcast(
    buffer: torch.Tensor,
    src_rank: int,
    dst_ranks: tuple[int, ...],
) -> dist.Work:
    """Execute broadcast operation.

    Args:
        buffer: Data buffer to broadcast
        src_rank: Source rank that broadcasts
        dst_ranks: Destination ranks

    Returns:
        Work handle for async operation
    """
    group_ranks = sorted([src_rank, *dst_ranks])
    group = get_or_create_process_group(group_ranks)
    return dist.broadcast(buffer, src=src_rank, group=group, async_op=True)


def execute_transfer(
    buffer: torch.Tensor,
    transfer_type: TransferType,
    is_source: bool,
    src_rank: int,
    dst_ranks: tuple[int, ...],
) -> dist.Work | None:
    """Execute transfer operation based on type.

    Unified entry point for all transfer types.

    Returns:
        Work handle for async operations, None for SELF_COPY.
    """
    match transfer_type:
        case TransferType.SELF_COPY:
            return None
        case TransferType.P2P:
            if is_source:
                return execute_p2p_send(buffer, dst_ranks[0])
            else:
                return execute_p2p_recv(buffer, src_rank)
        case TransferType.BROADCAST:
            return execute_broadcast(buffer, src_rank, dst_ranks)
