"""Communication Executor - Execute transfer operations."""

import torch.distributed as dist

from .ir import SourceChunk, TargetChunk, TransferType
from .utils import get_or_create_process_group


def _prepare_send_buffer(chunk: SourceChunk) -> None:
    """Prepare send buffer for a single SourceChunk.

    Args:
        chunk: SourceChunk to prepare (must have .tensor bound)
    """
    match chunk.transfer_type:
        case TransferType.SELF_COPY:
            pass
        case _:
            if chunk.tensor is None:
                raise ValueError("SourceChunk has no tensor bound. Call bind_tensors_to_chunks() first.")
            chunk.buffer = chunk.tensor[chunk.slice_tuples].contiguous()


def _prepare_recv_buffer(chunk: TargetChunk) -> None:
    """Prepare receive buffer for a single TargetChunk.

    Args:
        chunk: TargetChunk to prepare.
    """
    if chunk.tensor is None:
        raise ValueError("TargetChunk has no tensor bound. Call bind_tensors_to_chunks() first.")

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
            group_ranks = sorted([chunk.src_rank] + chunk.dst_ranks)
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
            # group_key is (src_rank, tuple(sorted(dst_ranks))); construct subgroup in sorted order
            src_rank, dst_tuple = chunk.group_key
            group_ranks = sorted([src_rank] + list(dst_tuple))
            group = get_or_create_process_group(group_ranks)
            chunk.work = dist.broadcast(tensor=chunk.buffer, src=src_rank, group=group, async_op=True)
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


def _prepare_chunk(chunk: SourceChunk | TargetChunk) -> None:
    """Prepare buffer for a chunk (polymorphic)."""
    if isinstance(chunk, SourceChunk):
        _prepare_send_buffer(chunk)
    else:
        _prepare_recv_buffer(chunk)


def _launch_chunk(chunk: SourceChunk | TargetChunk) -> None:
    """Launch async operation for a chunk (polymorphic)."""
    if isinstance(chunk, SourceChunk):
        _launch_send(chunk)
    else:
        _launch_recv(chunk)


def _cleanup_chunk(chunk: SourceChunk | TargetChunk) -> None:
    """Cleanup and assemble chunk (polymorphic)."""
    if isinstance(chunk, SourceChunk):
        _cleanup_send_chunk(chunk)
    else:
        _assemble_chunk(chunk)


def _is_complete(chunk: SourceChunk | TargetChunk) -> bool:
    """Check if async operation is complete."""
    if chunk.work is None:
        return True
    return chunk.work.is_completed()


def execute_pipeline(
    chunks: list[SourceChunk | TargetChunk],
    max_in_flight: int = 4,
) -> None:
    """Execute chunks with polling-based producer-consumer pipeline.

    Three queues:
    - candidate: chunks waiting to be prepared
    - prepared: chunks with buffers ready, waiting to launch
    - in_flight: chunks with async operations running

    Constraint: len(prepared) + len(in_flight) <= max_in_flight

    This enables true overlap between buffer preparation and async operations
    without relying on stage_id grouping.

    Args:
        chunks: Unified list of SourceChunk and TargetChunk operations
        max_in_flight: Maximum combined size of prepared + in_flight queues
    """
    if not chunks:
        return

    # Initialize three queues
    candidate = chunks.copy()  # shallow copy
    prepared: list[SourceChunk | TargetChunk] = []
    in_flight: list[SourceChunk | TargetChunk] = []

    # Pre-fill prepared queue up to max_in_flight
    while candidate and len(prepared) < max_in_flight:
        chunk = candidate.pop(0)
        _prepare_chunk(chunk)
        prepared.append(chunk)

    # Main polling loop
    while prepared or in_flight or candidate:
        # Launch prepared chunks if there's room in in_flight
        while prepared:
            chunk = prepared.pop(0)
            _launch_chunk(chunk)
            in_flight.append(chunk)

        # Poll in_flight for completions (non-blocking)
        completed_indices = []
        for i, chunk in enumerate(in_flight):
            if _is_complete(chunk):
                completed_indices.append(i)

        # Clean up completed chunks (iterate in reverse to maintain indices)
        for i in reversed(completed_indices):
            chunk = in_flight.pop(i)
            _cleanup_chunk(chunk)

        # Prepare more chunks if we have space
        while candidate and len(prepared) < max_in_flight:
            chunk = candidate.pop(0)
            _prepare_chunk(chunk)
            prepared.append(chunk)

        # If nothing completed and we still have work, wait for at least one to complete
        if not completed_indices and in_flight:
            # Block on the first in-flight chunk
            chunk = in_flight[0]
            if chunk.work is not None:
                chunk.work.wait()
            # Will be cleaned up in next iteration
