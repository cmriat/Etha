"""Group chunks into buckets for coalesced transfer."""

from collections import defaultdict

from .ir import Chunk, Bucket


def _make_bucket(chunks: list[Chunk]) -> Bucket:
    return Bucket(chunks=chunks, device=chunks[0].tensor.device)


def chunk_to_bucket_ops(
    chunks: list[Chunk],
    bucket_size: int,
) -> list[Bucket]:
    """Bundle same-route chunks (shared ``bucket_key``) up to ``bucket_size`` bytes.

    A chunk at least ``bucket_size`` becomes its own bucket; ``bucket_size=1``
    therefore gives one chunk per bucket (no coalescing). A ``no_coalesce`` chunk
    (Partial transfer) also stays single-entry — see ``Chunk.no_coalesce``.
    """
    buckets: list[Bucket] = []
    grouped_state: dict[tuple, tuple[list[Chunk], int]] = defaultdict(lambda: ([], 0))

    for chunk in chunks:
        chunk_bytes = chunk.nbytes
        if chunk_bytes >= bucket_size or chunk.no_coalesce:
            buckets.append(_make_bucket([chunk]))
            continue

        key = chunk.bucket_key
        current_chunks, current_bytes = grouped_state[key]
        current_chunks.append(chunk)
        current_bytes += chunk_bytes
        if current_bytes >= bucket_size:
            buckets.append(_make_bucket(current_chunks))
            grouped_state[key] = ([], 0)
        else:
            grouped_state[key] = (current_chunks, current_bytes)

    for current_chunks, _ in grouped_state.values():
        if current_chunks:
            buckets.append(_make_bucket(current_chunks))
    return buckets
