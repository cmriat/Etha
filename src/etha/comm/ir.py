"""Intermediate Representation for tensor transfer operations."""

import logging
from enum import Enum
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


class TransferType(Enum):
    """Transfer operation types."""

    SELF_COPY = "self_copy"  # Local copy within same rank
    P2P = "p2p"  # Point-to-point transfer between two ranks
    BROADCAST = "broadcast"  # One-to-many transfer


@dataclass(slots=True, kw_only=True)
class BaseChunk:
    """Base class for transfer chunks."""

    # Shape info
    chunk_shape: tuple[int, ...]  # Shape of the data being transferred

    # Transfer method
    transfer_type: TransferType

    # Tensor reference (None during planning, populated during binding)
    tensor: torch.Tensor | None = None

    # Buffer management
    buffer: torch.Tensor | None = None

    # Async work handle (None for SELF_COPY or before launch, populated during execution)
    work: "torch.distributed.Work | None" = None

    slice_tuples: tuple[slice, ...] = ()  # Slice tuple for tensor indexing

    target_dtype: torch.dtype | None = None


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

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        return (
            f"SourceChunk("
            f"type={self.transfer_type.name[:3] if self.transfer_type else '???'}, "
            f"src={self.src_rank}→{self.dst_ranks}, "
            f"tensor={self.tensor is not None})"
        )


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

    src_slice_tuples: tuple[slice, ...] = ()  # Slice tuple for source tensor (self_copy only)

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        return (
            f"TargetChunk("
            f"type={self.transfer_type.name[:3] if self.transfer_type else '???'}, "
            f"src={self.src_rank}→dst={self.dst_rank}, "
            f"tensor={self.tensor is not None})"
        )


@dataclass(slots=True, kw_only=True)
class Bucket:
    """Bucket for transfer operations."""

    transfer_type: TransferType
    is_source: bool
    dst_ranks: tuple[int, ...] | None = None
    src_rank: int | None = None
    group_key: tuple[int, tuple[int, ...]] | None = None
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    buffer: torch.Tensor | None = None
    work: "torch.distributed.Work | None" = None
    buffer_ready_event: torch.cuda.Event | None = None
    offsets: list[tuple[int, int, BaseChunk, tuple[int, ...]]]

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        kind = "src" if self.is_source else "dst"
        chunk_count = len(self.offsets)
        return f"Bucket({kind}, chunks={chunk_count} src_rank={self.src_rank}→dst_ranks={self.dst_ranks})"
