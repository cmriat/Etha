"""Communication Executor - Execute transfer operations."""

import torch.distributed as dist

from .utils import get_or_create_process_group
from .chunk_ops import SourceChunk, TargetChunk, TransferType


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
                raise ValueError(
                    f"SourceChunk {chunk.chunk_id} has no tensor bound. Call bind_tensors_to_chunks() first."
                )
            chunk.buffer = chunk.tensor[chunk.slice_tuples].contiguous()


def _prepare_recv_buffer(chunk: TargetChunk) -> None:
    """Prepare receive buffer for a single TargetChunk.

    Args:
        chunk: TargetChunk to prepare.
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
    # Sort chunks by (src_rank, dst_rank) to ensure consistent order across all ranks
    # This prevents deadlocks in P2P communication
    sorted_source_chunks = sorted(
        source_chunks, key=lambda c: (c.src_rank, min(c.dst_ranks) if c.dst_ranks else c.src_rank)
    )
    sorted_target_chunks = sorted(target_chunks, key=lambda c: (c.src_rank, c.dst_rank))

    # === Phase 1: Prepare buffers ===
    for chunk in sorted_source_chunks:
        _prepare_send_buffer(chunk)

    for chunk in sorted_target_chunks:
        _prepare_recv_buffer(chunk)

    # === Phase 2: Launch all async operations ===
    # Important: Launch all sends first, then all receives to avoid deadlock
    for chunk in sorted_source_chunks:
        _launch_send(chunk)

    for chunk in sorted_target_chunks:
        _launch_recv(chunk)

    # === Phase 3: Wait for all receives first, then sends ===
    # This ensures receivers are ready before senders complete
    for chunk in sorted_target_chunks:
        _assemble_chunk(chunk)

    for chunk in sorted_source_chunks:
        _cleanup_send_chunk(chunk)


def execute_pipelined(
    source_chunks: list[SourceChunk],
    target_chunks: list[TargetChunk],
    max_in_flight: int = 2,
) -> None:
    """Execute transfer with pipelining based on chunk stage IDs.

    Chunks are grouped by their stage_id field (assigned during planning).
    Multiple stages can be in-flight simultaneously, with max_in_flight
    controlling the pipeline depth for overlapped communication.

    Args:
        source_chunks: Chunks to send (must have .tensor bound and .stage_id set)
        target_chunks: Chunks to receive (must have .tensor bound and .stage_id set)
        max_in_flight: Maximum number of stages in-flight simultaneously

    Note:
        stage_id is assigned during map_to_chunk_ops() via chunks_per_stage parameter.
        Paired send/recv operations are guaranteed to be in the same stage.
    """
    from collections import defaultdict

    # Group chunks by stage_id
    src_by_stage = defaultdict(list)
    for chunk in source_chunks:
        src_by_stage[chunk.stage_id].append(chunk)

    tgt_by_stage = defaultdict(list)
    for chunk in target_chunks:
        tgt_by_stage[chunk.stage_id].append(chunk)

    # Get all stage IDs (sorted for deterministic execution)
    all_stages = sorted(set(src_by_stage.keys()) | set(tgt_by_stage.keys()))

    # Pipeline state: [(stage_id, src_chunks, tgt_chunks), ...]
    in_flight_stages = []

    for stage_id in all_stages:
        src_stage = src_by_stage.get(stage_id, [])
        tgt_stage = tgt_by_stage.get(stage_id, [])

        # === Phase 1: Prepare buffers ===
        for chunk in src_stage:
            _prepare_send_buffer(chunk)
        for chunk in tgt_stage:
            _prepare_recv_buffer(chunk)

        # === Phase 2: Launch async operations ===
        for chunk in src_stage:
            _launch_send(chunk)
        for chunk in tgt_stage:
            _launch_recv(chunk)

        # Add to in-flight queue
        in_flight_stages.append((stage_id, src_stage, tgt_stage))

        # === Phase 3: Pipeline control - wait for oldest if pipeline is full ===
        if len(in_flight_stages) >= max_in_flight:
            # Wait for oldest stage to complete
            old_stage_id, old_src, old_tgt = in_flight_stages.pop(0)

            for chunk in old_src:
                _cleanup_send_chunk(chunk)
            for chunk in old_tgt:
                _assemble_chunk(chunk)

    # === Phase 4: Drain pipeline - wait for all remaining stages ===
    for _, src_stage, tgt_stage in in_flight_stages:
        for chunk in src_stage:
            _cleanup_send_chunk(chunk)
        for chunk in tgt_stage:
            _assemble_chunk(chunk)
