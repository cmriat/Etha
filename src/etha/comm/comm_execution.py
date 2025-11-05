"""Communication Executor - Execute transfer operations."""

import torch
import torch.distributed as dist

from .utils import get_or_create_process_group
from .chunk_ops import SourceChunk, TargetChunk, TransferType


def _prepare_send_buffer(chunk: SourceChunk, local_tensor: torch.Tensor) -> None:
    """Prepare send buffer for a single SourceChunk.

    Args:
        chunk: SourceChunk to prepare
        local_tensor: Source tensor to slice from
    """
    match chunk.transfer_type:
        case TransferType.SELF_COPY:
            pass
        case _:
            data_slice = local_tensor[chunk.slice_tuples]
            chunk.buffer = data_slice.contiguous()


def _prepare_recv_buffer(
    chunk: TargetChunk,
    source_local_tensor: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Prepare receive buffer for a single TargetChunk.

    Args:
        chunk: TargetChunk to prepare
        source_local_tensor: Source tensor for self-copy (can be None)
        device: Device to allocate buffers on
        dtype: Data type for buffers
    """
    match chunk.transfer_type:
        case TransferType.SELF_COPY:
            chunk.buffer = source_local_tensor[chunk.src_slice_tuples]
        case _:
            chunk.buffer = torch.empty(chunk.chunk_shape, dtype=dtype, device=device)


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
            group_ranks = [chunk.src_rank] + list(chunk.group_key[1]) if chunk.group_key else [chunk.src_rank]
            group = get_or_create_process_group(group_ranks)
            return dist.broadcast(tensor=chunk.buffer, src=chunk.src_rank, group=group, async_op=True)
        case TransferType.P2P:
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
    if chunk.buffer is None:
        return
    final_tensor[chunk.slice_tuples].copy_(chunk.buffer)
    chunk.buffer = None  # Free buffer reference


def execute_naive(
    source_chunks: list[SourceChunk],
    target_chunks: list[TargetChunk],
    source_local_tensor: torch.Tensor | None,
    target_local_tensor: torch.Tensor | None,
) -> None:
    """Execute transfer in naive mode: prepare → launch all → wait all → assemble.

    Pure function - all state is in parameters or local variables.

    Args:
        source_chunks: Chunks to send
        target_chunks: Chunks to receive
        source_local_tensor: Source tensor for slicing and self-copy
        target_local_tensor: Final assembled tensor (can be None for sender-only)
    """
    if source_local_tensor is None and target_local_tensor is None:
        raise ValueError("Both source_local_tensor and target_local_tensor are None")
    if target_local_tensor is not None:
        device = target_local_tensor.device
        dtype = target_local_tensor.dtype
    if source_local_tensor is not None:
        device = source_local_tensor.device
        dtype = source_local_tensor.dtype

    # === Phase 1: Prepare buffers ===
    for chunk in source_chunks:
        _prepare_send_buffer(chunk, source_local_tensor)

    for chunk in target_chunks:
        _prepare_recv_buffer(chunk, source_local_tensor, device, dtype)

    # === Phase 2: Launch sends ===
    send_works: dict[int, dist.Work] = {}

    for chunk in source_chunks:
        work = _launch_send(chunk)
        send_works[chunk.chunk_id] = work

    # === Phase 3: Launch receives ===
    recv_works: dict[int, dist.Work] = {}

    for chunk in target_chunks:
        match chunk.transfer_type:
            case TransferType.SELF_COPY:
                continue
            case _:
                work = _launch_recv(chunk)
                recv_works[chunk.chunk_id] = work

    # === Phase 4: Wait all ===
    for work in recv_works.values():
        work.wait()
    for work in send_works.values():
        work.wait()

    # === Phase 5: Assemble final tensor ===
    if target_local_tensor is not None:
        for chunk in target_chunks:
            _assemble_chunk(chunk, target_local_tensor)
