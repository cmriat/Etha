"""Intermediate Representation for tensor transfer operations."""

from typing import Literal
from dataclasses import dataclass

import torch


@dataclass(slots=True, kw_only=True)
class BaseChunk:
    """Base class for transfer chunks.

    Shared fields for both send and receive operations.
    """

    # Identity
    chunk_id: int  # Unique ID for this chunk

    # Shape info
    chunk_shape: tuple[int, ...]  # Shape of the data being transferred

    # Transfer method
    transfer_type: Literal["self_copy", "p2p", "broadcast"]

    # Buffer management
    buffer: torch.Tensor | None = None


@dataclass(slots=True, kw_only=True)
class SourceChunk(BaseChunk):
    """Source-side transfer chunk (send operations).

    Represents a chunk of data to be sent from source rank to one or more target ranks.
    """

    # Source info
    src_rank: int  # Rank that owns this data
    src_idx: tuple  # Multi-dimensional index in source tensor

    # Destination info
    dst_ranks: list[int]  # Target ranks (len > 1 triggers broadcast)

    # Broadcast info (None for self_copy and p2p)
    group_key: tuple[int, tuple[int, ...]] | None = None  # (src_rank, tuple(sorted(dst_ranks)))


@dataclass(slots=True, kw_only=True)
class TargetChunk(BaseChunk):
    """Target-side transfer chunk (receive + assemble operations).

    Represents a chunk of data to be received and assembled into final tensor.
    """

    # Target info
    dst_rank: int  # Rank that will receive this data
    dst_idx: tuple  # Multi-dimensional index in target tensor

    # Source info (where data comes from)
    src_rank: int
    src_idx: tuple  # Multi-dimensional index in source tensor

    # Broadcast info (None for self_copy and p2p)
    group_key: tuple[int, tuple[int, ...]] | None = None  # (src_rank, tuple(sorted(dst_ranks)))
