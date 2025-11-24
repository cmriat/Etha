"""Get buckets from chunks."""

import math
from collections import defaultdict

import torch

from .ir import Bucket, BaseChunk, BucketEntry


def _chunk_nbytes(chunk: BaseChunk) -> int:
    """Calculate chunk size in bytes."""
    dtype = chunk.target_dtype if chunk.target_dtype else chunk.tensor.dtype
    element_size = torch.empty((), dtype=dtype).element_size()
    return math.prod(chunk.chunk_shape) * element_size


def _calculate_bucket_entries(
    grouped_chunks: list[BaseChunk],
) -> list[BucketEntry]:
    """Calculate bucket entries with byte-based offsets."""
    entries: list[BucketEntry] = []
    cursor = 0
    for chunk in grouped_chunks:
        nbytes = _chunk_nbytes(chunk)
        entries.append(BucketEntry(offset=cursor, nbytes=nbytes, chunk=chunk))
        cursor += nbytes
    return entries


def _build_bucket(
    entries: list[BucketEntry],
) -> Bucket:
    first_chunk = entries[0].chunk
    device = first_chunk.tensor.device
    transfer_type = first_chunk.transfer_type
    dst_ranks = first_chunk.dst_ranks
    src_rank = first_chunk.src_rank
    total_bytes = entries[-1].offset + entries[-1].nbytes
    key = first_chunk.bucket_key
    return Bucket(
        transfer_type=transfer_type,
        is_source=first_chunk.is_source,
        dst_ranks=dst_ranks,
        src_rank=src_rank,
        device=device,
        key=key,
        total_bytes=total_bytes,
        entries=entries,
    )


def chunk_to_bucket_ops(
    chunks: list[BaseChunk],
    bucket_size: int,
) -> list[Bucket]:
    buckets: list[Bucket] = []
    grouped_state: dict[tuple, tuple[list[BaseChunk], int]] = defaultdict(lambda: ([], 0))

    for chunk in chunks:
        chunk_bytes = _chunk_nbytes(chunk)
        if chunk_bytes >= bucket_size:
            entries = _calculate_bucket_entries([chunk])
            buckets.append(_build_bucket(entries))
            continue

        key = chunk.bucket_key
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
