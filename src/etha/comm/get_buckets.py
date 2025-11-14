"""Get buckets from chunks."""

import math
from collections import defaultdict

import torch

from .ir import Bucket, BucketEntry, SourceChunk, TargetChunk


def _chunk_nbytes(chunk: SourceChunk | TargetChunk) -> int:
    dtype = chunk.target_dtype if isinstance(chunk, SourceChunk) else chunk.tensor.dtype
    element_size = torch.empty((), dtype=dtype).element_size()
    return math.prod(chunk.chunk_shape) * element_size


def _bucket_key(chunk: SourceChunk | TargetChunk) -> tuple:
    if isinstance(chunk, SourceChunk):
        return chunk.src_idx
    return chunk.dst_idx


def _calculate_bucket_entries(
    grouped_chunks: list[SourceChunk | TargetChunk],
) -> list[BucketEntry]:
    entries: list[BucketEntry] = []
    cursor = 0
    for chunk in grouped_chunks:
        numel = math.prod(chunk.chunk_shape)
        entries.append(BucketEntry(offset=cursor, numel=numel, chunk=chunk))
        cursor += numel
    return entries


def _build_bucket(
    entries: list[BucketEntry],
) -> Bucket:
    first_chunk = entries[0].chunk
    is_source = isinstance(first_chunk, SourceChunk)
    dtype = first_chunk.target_dtype if is_source else first_chunk.tensor.dtype
    device = first_chunk.tensor.device
    transfer_type = first_chunk.transfer_type
    dst_ranks = first_chunk.dst_ranks
    src_rank = first_chunk.src_rank
    total_elems = entries[-1].offset + entries[-1].numel
    key = _bucket_key(first_chunk)
    return Bucket(
        transfer_type=transfer_type,
        is_source=is_source,
        dst_ranks=dst_ranks,
        src_rank=src_rank,
        dtype=dtype,
        device=device,
        key=key,
        total_elems=total_elems,
        entries=entries,
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
            entries = _calculate_bucket_entries([chunk])
            buckets.append(_build_bucket(entries))
            continue

        key = _bucket_key(chunk)
        current_chunks, current_bytes = grouped_state[key]
        current_chunks.append(chunk)
        current_bytes += chunk_bytes
        if current_bytes >= bucket_size:
            entries = _calculate_bucket_entries(current_chunks)
            buckets.append(_build_bucket(entries))
            grouped_state[key] = ([], 0)
        else:
            grouped_state[key] = (current_chunks, current_bytes)

    for current_chunks, _ in grouped_state.values():
        if not current_chunks:
            continue
        entries = _calculate_bucket_entries(current_chunks)
        buckets.append(_build_bucket(entries))
    return buckets
