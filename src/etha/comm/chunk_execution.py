"""Communication Executor."""

import torch

from .ir import BaseChunk


def execute_chunk_simple(
    chunks: list[BaseChunk],
) -> None:
    """Execute chunks using polymorphic prepare/launch/finalize methods.

    This is the simple sequential execution strategy - each chunk is fully
    prepared, launched, and finalized before moving to the next chunk.
    """
    for chunk in chunks:
        if chunk.tensor is None:
            continue
        chunk.prepare()
        chunk.launch()
        chunk.finalize()
        torch.cuda.synchronize()
