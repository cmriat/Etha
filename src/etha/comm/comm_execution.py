"""Communication Executor - Execute transfer operations."""

import torch
import torch.distributed as dist

from etha.comm.utils import get_or_create_process_group

from .utils import (
    get_slicer_tuples,
    get_slice_from_multi_index,
    get_or_create_process_group,
)
from .chunk_ir import SourceChunk, TargetChunk


def prepare_send_buffers(
    chunks: list[SourceChunk],
    local_tensor: torch.Tensor,
    source_num_slicers: list[int],
) -> None:
    """Prepare send buffers by slicing source tensor.

    Modifies chunks in-place by setting chunk.buffer.

    Args:
        chunks: SourceChunks to prepare
        local_tensor: Source tensor to slice from
        source_num_slicers: Partitioning of source tensor
    """
    if not chunks or local_tensor is None:
        return

    # Extend num_slicers to match tensor dimensions
    source_num_slicers = source_num_slicers + [1] * (local_tensor.ndim - len(source_num_slicers))

    # Pre-compute all slice tuples for efficiency
    slicer_tuples = get_slicer_tuples(local_tensor.shape, source_num_slicers)

    for chunk in chunks:
        if chunk.transfer_type == "self_copy":
            # Self-copy doesn't need send buffer
            continue

        # Slice source tensor
        slice_tuple = get_slice_from_multi_index(chunk.src_idx, source_num_slicers, slicer_tuples)
        data_slice = local_tensor[slice_tuple]

        # Make contiguous for network transfer
        chunk.buffer = data_slice if data_slice.is_contiguous() else data_slice.contiguous()

        # Update chunk_shape if it was a placeholder
        if not chunk.chunk_shape:
            chunk.chunk_shape = tuple(chunk.buffer.shape)


def prepare_recv_buffers(
    chunks: list[TargetChunk],
    source_local_tensor: torch.Tensor | None,
    source_num_slicers: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Prepare receive buffers by allocating empty tensors.

    Modifies chunks in-place by setting chunk.buffer.

    Args:
        chunks: TargetChunks to prepare
        source_local_tensor: Source tensor for self-copy (can be None)
        source_num_slicers: Partitioning of source tensor (for self-copy)
        device: Device to allocate buffers on
        dtype: Data type for buffers
    """
    # Pre-compute slicer tuples for self-copy
    slicer_tuples = None
    if source_local_tensor is not None:
        source_num_slicers_extended = source_num_slicers + [1] * (source_local_tensor.ndim - len(source_num_slicers))
        slicer_tuples = get_slicer_tuples(source_local_tensor.shape, source_num_slicers_extended)

    for chunk in chunks:
        if chunk.transfer_type == "self_copy":
            # Self-copy: slice from source tensor
            if source_local_tensor is not None and slicer_tuples is not None:
                slice_tuple = get_slice_from_multi_index(chunk.src_idx, source_num_slicers_extended, slicer_tuples)
                chunk.buffer = source_local_tensor[slice_tuple]
            continue

        # Allocate empty buffer for network receive
        chunk.buffer = torch.empty(chunk.chunk_shape, dtype=dtype, device=device)


def execute_naive(
    source_chunks: list[SourceChunk],
    target_chunks: list[TargetChunk],
    target_tensor_shape: tuple[int, ...],
    target_num_slicers: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Execute transfer in naive mode: launch all → wait all → assemble.

    Pure function - all state is in parameters or local variables.

    Args:
        source_chunks: Chunks to send (with buffers prepared)
        target_chunks: Chunks to receive (with buffers prepared)
        target_tensor_shape: Shape of final assembled tensor
        target_num_slicers: Partitioning of target tensor
        device: Device for final tensor
        dtype: Data type for final tensor

    Returns:
        Assembled target tensor
    """
    # === Phase 1: Launch sends ===
    send_works: dict[int, dist.Work] = {}

    for chunk in source_chunks:
        work = _launch_send(chunk)
        send_works[chunk.chunk_id] = work

    # === Phase 2: Launch receives ===
    recv_works: dict[int, dist.Work] = {}

    for chunk in target_chunks:
        if chunk.transfer_type == "self_copy":
            continue  # Self-copy already handled in prepare phase

        work = _launch_recv(chunk)
        recv_works[chunk.chunk_id] = work

    # === Phase 3: Wait all ===
    for work in recv_works.values():
        work.wait()
    for work in send_works.values():
        work.wait()

    # === Phase 4: Assemble final tensor ===
    final_tensor = torch.empty(target_tensor_shape, device=device, dtype=dtype)

    # Extend num_slicers to match tensor dimensions
    target_num_slicers = target_num_slicers + [1] * (len(target_tensor_shape) - len(target_num_slicers))

    for chunk in target_chunks:
        _assemble_chunk(chunk, final_tensor, target_num_slicers)

    return final_tensor


def _launch_send(chunk: SourceChunk) -> dist.Work:
    """Launch async send operation for a SourceChunk.

    Args:
        chunk: SourceChunk with buffer prepared

    Returns:
        Work handle for async operation
    """
    if chunk.transfer_type == "broadcast":
        # Broadcast to multiple ranks
        group_ranks = [chunk.src_rank] + chunk.dst_ranks
        group = get_or_create_process_group(group_ranks)
        return dist.broadcast(tensor=chunk.buffer, src=chunk.src_rank, group=group, async_op=True)
    elif chunk.transfer_type == "p2p":
        # P2P send to single rank
        assert len(chunk.dst_ranks) == 1, f"P2P should have exactly 1 dst_rank, got {len(chunk.dst_ranks)}"
        # Note: batch_isend_irecv is used in original code, but here we do individual sends
        # This will be batched at a higher level if needed
        return dist.isend(tensor=chunk.buffer, dst=chunk.dst_ranks[0])
    else:
        raise ValueError(f"Unknown transfer type for send: {chunk.transfer_type}")


def _launch_recv(chunk: TargetChunk) -> dist.Work:
    """Launch async receive operation for a TargetChunk.

    Args:
        chunk: TargetChunk with buffer allocated

    Returns:
        Work handle for async operation
    """
    if chunk.transfer_type == "broadcast":
        # Receive broadcast from source rank
        group_ranks = [chunk.src_rank] + list(chunk.group_key[1]) if chunk.group_key else [chunk.src_rank]
        group = get_or_create_process_group(group_ranks)
        return dist.broadcast(tensor=chunk.buffer, src=chunk.src_rank, group=group, async_op=True)
    elif chunk.transfer_type == "p2p":
        # P2P receive from source rank
        return dist.irecv(tensor=chunk.buffer, src=chunk.src_rank)
    else:
        raise ValueError(f"Unknown transfer type for recv: {chunk.transfer_type}")


def _assemble_chunk(
    chunk: TargetChunk,
    final_tensor: torch.Tensor,
    target_num_slicers: list[int],
) -> None:
    """Assemble a chunk into the final tensor.

    Args:
        chunk: TargetChunk to assemble
        final_tensor: Destination tensor
        target_num_slicers: Partitioning of target tensor
    """
    # Buffer should already be set (either from network recv or self-copy)
    if chunk.buffer is None:
        return

    # Calculate slice ranges for target position
    slice_ranges = []
    for dim, coord in enumerate(chunk.dst_idx):
        if target_num_slicers[dim] > 1:
            slice_size = final_tensor.shape[dim] // target_num_slicers[dim]
            start = coord * slice_size
            end = start + slice_size
            slice_ranges.append(slice(start, end))
        else:
            slice_ranges.append(slice(None))

    # Copy buffer to final tensor
    final_tensor[tuple(slice_ranges)].copy_(chunk.buffer)
