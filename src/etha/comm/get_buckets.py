"""Get buckets from chunks."""

import math
from collections import defaultdict

import torch

from .ir import Bucket, SourceChunk, TargetChunk


def _chunk_nbytes(chunk: SourceChunk | TargetChunk) -> int:
    dtype = chunk.target_dtype if isinstance(chunk, SourceChunk) else chunk.tensor.dtype
    element_size = torch.empty((), dtype=dtype).element_size()
    return math.prod(chunk.chunk_shape) * element_size


def _bucket_key(chunk: SourceChunk | TargetChunk) -> tuple:
    if isinstance(chunk, SourceChunk):
        return chunk.src_idx
    return chunk.dst_idx


def _calculate_bucket_offsets(
    grouped_chunks: list[SourceChunk | TargetChunk],
) -> list[tuple[int, int, SourceChunk | TargetChunk, tuple[int, ...]]]:
    offsets: list[tuple[int, int, SourceChunk | TargetChunk, tuple[int, ...]]] = []
    cursor = 0
    for chunk in grouped_chunks:
        numel = math.prod(chunk.chunk_shape)
        shape = tuple(int(dim) for dim in chunk.chunk_shape)
        offsets.append((cursor, numel, chunk, shape))
        cursor += numel
    return offsets


def _build_bucket(
    offsets: list[tuple[int, int, SourceChunk | TargetChunk, tuple[int, ...]]],
) -> Bucket:
    first_chunk = offsets[0][2]
    is_source = isinstance(first_chunk, SourceChunk)
    dtype = first_chunk.target_dtype if is_source else first_chunk.tensor.dtype
    device = first_chunk.tensor.device
    transfer_type = first_chunk.transfer_type
    dst_ranks = tuple(first_chunk.dst_ranks) if is_source else None
    src_rank = first_chunk.src_rank
    return Bucket(
        transfer_type=transfer_type,
        is_source=is_source,
        dst_ranks=dst_ranks,
        src_rank=src_rank,
        group_key=first_chunk.group_key,
        dtype=dtype,
        device=device,
        offsets=offsets,
    )


def chunk_to_bucket_ops(
    chunks: list[SourceChunk | TargetChunk],
    bucket_size: int,
) -> list[Bucket]:
    buckets: list[Bucket] = []
    grouped_state: dict[tuple, tuple[list[SourceChunk | TargetChunk], int]] = defaultdict(lambda: ([], 0))
    for chunk in chunks:
        chunk_bytes = _chunk_nbytes(chunk)
        if chunk_bytes >= bucket_size:
            offsets = _calculate_bucket_offsets([chunk])
            buckets.append(_build_bucket(offsets))
            continue
        key = _bucket_key(chunk)
        current_chunks, current_bytes = grouped_state[key]
        current_chunks.append(chunk)
        current_bytes += chunk_bytes
        if current_bytes >= bucket_size:
            offsets = _calculate_bucket_offsets(current_chunks)
            buckets.append(_build_bucket(offsets))
            grouped_state[key] = ([], 0)
        else:
            grouped_state[key] = (current_chunks, current_bytes)
    for current_chunks, _ in grouped_state.values():
        if not current_chunks:
            continue
        offsets = _calculate_bucket_offsets(current_chunks)
        buckets.append(_build_bucket(offsets))
    return buckets
