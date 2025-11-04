"""Communication Executor - Execute transfer operations."""

import torch
import torch.distributed as dist

from .utils import get_or_create_process_group
from .chunk_ir import SourceChunk, TargetChunk, TransferType


def prepare_send_buffers(
    chunks: list[SourceChunk],
    local_tensor: torch.Tensor,
) -> None:
    """Prepare send buffers by slicing source tensor.

    Modifies chunks in-place by setting chunk.buffer.

    Args:
        chunks: SourceChunks to prepare
        local_tensor: Source tensor to slice from
    """
    if not chunks or local_tensor is None:
        return

    for chunk in chunks:
        match chunk.transfer_type:
            case TransferType.SELF_COPY:
                continue  # Self-copy doesn't need send buffer
            case _:
                data_slice = local_tensor[chunk.slice_tuples]
                chunk.buffer = data_slice.contiguous()


def prepare_recv_buffers(
    chunks: list[TargetChunk],
    source_local_tensor: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Prepare receive buffers by allocating empty tensors.

    Modifies chunks in-place by setting chunk.buffer.

    Args:
        chunks: TargetChunks to prepare
        source_local_tensor: Source tensor for self-copy (can be None)
        device: Device to allocate buffers on
        dtype: Data type for buffers
    """
    for chunk in chunks:
        match chunk.transfer_type:
            case TransferType.SELF_COPY:
                # Self-copy: slice from source tensor
                if source_local_tensor is not None:
                    chunk.buffer = source_local_tensor[chunk.src_slice_tuples]
            case _:
                # Allocate empty buffer for network receive
                chunk.buffer = torch.empty(chunk.chunk_shape, dtype=dtype, device=device)


def execute_naive(
    source_chunks: list[SourceChunk],
    target_chunks: list[TargetChunk],
    target_tensor: torch.Tensor | None,
) -> None:
    """Execute transfer in naive mode: launch all → wait all → assemble.

    Pure function - all state is in parameters or local variables.

    Args:
        source_chunks: Chunks to send (with buffers prepared)
        target_chunks: Chunks to receive (with buffers prepared)
        target_tensor: Final assembled tensor
    """
    # === Phase 1: Launch sends ===
    send_works: dict[int, dist.Work] = {}

    for chunk in source_chunks:
        work = _launch_send(chunk)
        send_works[chunk.chunk_id] = work

    # === Phase 2: Launch receives ===
    recv_works: dict[int, dist.Work] = {}

    for chunk in target_chunks:
        match chunk.transfer_type:
            case TransferType.SELF_COPY:
                continue  # Self-copy already handled in prepare phase
            case _:
                work = _launch_recv(chunk)
                recv_works[chunk.chunk_id] = work

    # === Phase 3: Wait all ===
    for work in recv_works.values():
        work.wait()
    for work in send_works.values():
        work.wait()

    # === Phase 4: Assemble final tensor ===
    if target_tensor is not None:
        for chunk in target_chunks:
            _assemble_chunk(chunk, target_tensor)


def _launch_send(chunk: SourceChunk) -> dist.Work:
    """Launch async send operation for a SourceChunk.

    Args:
        chunk: SourceChunk with buffer prepared

    Returns:
        Work handle for async operation
    """
    match chunk.transfer_type:
        case TransferType.BROADCAST:
            # Broadcast to multiple ranks
            group_ranks = [chunk.src_rank] + chunk.dst_ranks
            group = get_or_create_process_group(group_ranks)
            return dist.broadcast(tensor=chunk.buffer, src=chunk.src_rank, group=group, async_op=True)
        case TransferType.P2P:
            # P2P send to single rank
            assert len(chunk.dst_ranks) == 1, f"P2P should have exactly 1 dst_rank, got {len(chunk.dst_ranks)}"
            return dist.isend(tensor=chunk.buffer, dst=chunk.dst_ranks[0])
        case _:
            raise ValueError(f"Unknown transfer type for send: {chunk.transfer_type}")


def _launch_recv(chunk: TargetChunk) -> dist.Work:
    """Launch async receive operation for a TargetChunk.

    Args:
        chunk: TargetChunk with buffer allocated

    Returns:
        Work handle for async operation
    """
    match chunk.transfer_type:
        case TransferType.BROADCAST:
            # Receive broadcast from source rank
            group_ranks = [chunk.src_rank] + list(chunk.group_key[1]) if chunk.group_key else [chunk.src_rank]
            group = get_or_create_process_group(group_ranks)
            return dist.broadcast(tensor=chunk.buffer, src=chunk.src_rank, group=group, async_op=True)
        case TransferType.P2P:
            # P2P receive from source rank
            return dist.irecv(tensor=chunk.buffer, src=chunk.src_rank)
        case _:
            raise ValueError(f"Unknown transfer type for recv: {chunk.transfer_type}")


def _assemble_chunk(
    chunk: TargetChunk,
    final_tensor: torch.Tensor,
) -> None:
    """Assemble a chunk into the final tensor.

    Args:
        chunk: TargetChunk to assemble
        final_tensor: Destination tensor
    """
    # Buffer should already be set (either from network recv or self-copy)
    if chunk.buffer is None:
        return
    final_tensor[chunk.slice_tuples].copy_(chunk.buffer)
