"""Communication Executor - Execute transfer operations."""

import torch.distributed as dist

from .utils import get_or_create_process_group
from .chunk_ops import SourceChunk, TargetChunk, TransferType


def _prepare_send_buffer(chunk: SourceChunk) -> None:
    """Prepare send buffer for a single SourceChunk.

    Args:
        chunk: SourceChunk to prepare (must have .tensor bound)
    """
    if chunk.tensor is None:
        raise ValueError(f"SourceChunk {chunk.chunk_id} has no tensor bound. Call bind_tensors_to_chunks() first.")

    match chunk.transfer_type:
        case TransferType.SELF_COPY:
            pass
        case _:
            chunk.buffer = chunk.tensor[chunk.slice_tuples].contiguous()


def _prepare_recv_buffer(chunk: TargetChunk) -> None:
    """Prepare receive buffer for a single TargetChunk.

    Args:
        chunk: TargetChunk to prepare (must have .tensor bound)
    """
    if chunk.tensor is None:
        raise ValueError(f"TargetChunk {chunk.chunk_id} has no tensor bound. Call bind_tensors_to_chunks() first.")

    match chunk.transfer_type:
        case TransferType.SELF_COPY:
            chunk.buffer = chunk.tensor[chunk.src_slice_tuples]
        case _:
            chunk.buffer = chunk.tensor[chunk.slice_tuples].contiguous()


def _launch_send(chunk: SourceChunk) -> None:
    """Launch async send operation, store work handle in chunk.

    Args:
        chunk: SourceChunk with buffer prepared (modified in-place)
    """
    match chunk.transfer_type:
        case TransferType.SELF_COPY:
            chunk.work = None
        case TransferType.BROADCAST:
            group_ranks = [chunk.src_rank] + chunk.dst_ranks
            group = get_or_create_process_group(group_ranks)
            chunk.work = dist.broadcast(tensor=chunk.buffer, src=chunk.src_rank, group=group, async_op=True)
        case TransferType.P2P:
            assert len(chunk.dst_ranks) == 1, f"P2P should have exactly 1 dst_rank, got {len(chunk.dst_ranks)}"
            chunk.work = dist.isend(tensor=chunk.buffer, dst=chunk.dst_ranks[0])


def _launch_recv(chunk: TargetChunk) -> None:
    """Launch async receive operation, store work handle in chunk.

    Args:
        chunk: TargetChunk with buffer allocated (modified in-place)
    """
    match chunk.transfer_type:
        case TransferType.SELF_COPY:
            chunk.work = None
        case TransferType.BROADCAST:
            group_ranks = [chunk.src_rank] + list(chunk.group_key[1]) if chunk.group_key else [chunk.src_rank]
            group = get_or_create_process_group(group_ranks)
            chunk.work = dist.broadcast(tensor=chunk.buffer, src=chunk.src_rank, group=group, async_op=True)
        case TransferType.P2P:
            chunk.work = dist.irecv(tensor=chunk.buffer, src=chunk.src_rank)


def _assemble_chunk(chunk: TargetChunk) -> None:
    """Wait for recv to complete, assemble chunk into final tensor, and cleanup.

    Args:
        chunk: TargetChunk to assemble (must have .tensor bound)
    """
    # Wait for async recv to complete
    if chunk.work is not None:
        chunk.work.wait()
        chunk.work = None

    chunk.tensor[chunk.slice_tuples].copy_(chunk.buffer)
    chunk.buffer = None


def _cleanup_send_chunk(chunk: SourceChunk) -> None:
    """Wait for send to complete and cleanup buffer.

    Args:
        chunk: SourceChunk to cleanup
    """
    # Wait for async send to complete
    if chunk.work is not None:
        chunk.work.wait()
        chunk.work = None

    chunk.buffer = None


def execute_naive(
    source_chunks: list[SourceChunk],
    target_chunks: list[TargetChunk],
) -> None:
    """Execute transfer in naive mode: prepare → launch all → wait all → assemble.

    Pure function - all state is in chunk objects (including tensor references and work handles).

    IMPORTANT: Chunks must have tensor references bound via bind_tensors_to_chunks()
    before calling this function.

    Args:
        source_chunks: Chunks to send (must have .tensor bound)
        target_chunks: Chunks to receive (must have .tensor bound)

    Raises:
        ValueError: If chunks do not have tensor references bound
    """
    # TODO: may have mixed send/recv roles, should keep order to avoid deadlock
    # TODO: optimize with pipelining

    # === Phase 1: Prepare buffers ===
    for chunk in source_chunks:
        _prepare_send_buffer(chunk)

    for chunk in target_chunks:
        _prepare_recv_buffer(chunk)

    # === Phase 2: Launch all async operations ===
    for chunk in source_chunks:
        _launch_send(chunk)

    for chunk in target_chunks:
        _launch_recv(chunk)

    # === Phase 3: Wait + cleanup (each chunk completes its lifecycle) ===
    for chunk in source_chunks:
        _cleanup_send_chunk(chunk)

    for chunk in target_chunks:
        _assemble_chunk(chunk)
