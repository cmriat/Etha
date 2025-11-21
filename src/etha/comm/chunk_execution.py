"""Communication Executor."""

import torch

from .ir import BaseChunk, SendChunk
from .transfer_ops import execute_transfer


def execute_chunk_simple(
    chunks: list[BaseChunk],
) -> None:
    """Execute chunks using polymorphic prepare/finalize and transfer functions.

    This is the simple sequential execution strategy - each chunk is fully
    prepared, launched, and finalized before moving to the next chunk.
    """
    for chunk in chunks:
        if chunk.tensor is None:
            continue
        chunk.prepare()
        chunk.work = execute_transfer(
            chunk.buffer,
            chunk.transfer_type,
            isinstance(chunk, SendChunk),
            chunk.src_rank,
            chunk.dst_ranks,
        )
        chunk.finalize()
        torch.cuda.synchronize()
