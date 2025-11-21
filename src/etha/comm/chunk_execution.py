"""Communication Executor - Execute transfer operations."""

import torch
import torch.distributed as dist

from .ir import SourceChunk, TargetChunk, TransferType
from .utils import get_or_create_process_group


def _prepare_chunk(chunk: SourceChunk | TargetChunk, contiguous: bool = True) -> None:
    match chunk.transfer_type:
        case TransferType.SELF_COPY:
            if isinstance(chunk, TargetChunk):
                chunk.buffer = chunk.tensor[chunk.src_slice_tuples]
        case _:
            if contiguous:
                chunk.buffer = chunk.tensor[chunk.slice_tuples].contiguous()
            else:
                chunk.buffer = chunk.tensor[chunk.slice_tuples]
    if isinstance(chunk, SourceChunk) and chunk.target_dtype != chunk.buffer.dtype:
        chunk.buffer = chunk.buffer.to(chunk.target_dtype)


def _launch_chunk(chunk: SourceChunk | TargetChunk) -> None:
    match chunk.transfer_type:
        case TransferType.SELF_COPY:
            chunk.work = None
        case TransferType.BROADCAST:
            group_ranks = sorted([chunk.src_rank, *chunk.dst_ranks])
            group = get_or_create_process_group(group_ranks)
            chunk.work = dist.broadcast(chunk.buffer, src=chunk.src_rank, group=group, async_op=True)
        case TransferType.P2P:
            if isinstance(chunk, SourceChunk):
                chunk.work = dist.isend(chunk.buffer, dst=chunk.dst_ranks[0])
            else:
                chunk.work = dist.irecv(chunk.buffer, src=chunk.src_rank)


def _finalize_chunk(chunk: SourceChunk | TargetChunk) -> None:
    if chunk.work is not None:
        chunk.work.wait()
        chunk.work = None

    if isinstance(chunk, TargetChunk):
        chunk.tensor[chunk.slice_tuples].copy_(chunk.buffer, non_blocking=True)

    chunk.buffer = None


def execute_chunk_simple(
    chunks: list[SourceChunk | TargetChunk],
) -> None:
    for chunk in chunks:
        if chunk.tensor is None:
            continue
        _prepare_chunk(chunk)
        _launch_chunk(chunk)
        _finalize_chunk(chunk)
        torch.cuda.synchronize()
