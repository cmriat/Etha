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
class Transfer:
    """Intermediate representation for a send operation.

    Represents a logical transfer from one source to one or more destinations.
    Used during planning phase before converting to execution-ready Chunks.
    """

    src_rank: int  # Source rank
    src_idx: tuple  # Source multi-dimensional index
    dst_list: list[tuple[int, tuple]]  # [(dst_rank, dst_idx), ...]
    transfer_type: TransferType
    stage_id: int = 0  # Pipeline stage

    # Partitioning metadata for source and target tensors
    source_num_slicers: list[int]  # How source tensor is partitioned
    target_num_slicers: list[int]  # How target tensor is partitioned

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        dst_ranks = [r for r, _ in self.dst_list]
        type_name = self.transfer_type.name if self.transfer_type else "???"
        dst_summary = str(dst_ranks)
        return f"Transfer(type={type_name}, src={self.src_rank}→{dst_summary}, stage={self.stage_id})"


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

    # Pipeline stage ID (assigned during planning for pipelining support)
    stage_id: int = 0

    slice_tuples: tuple[slice, ...] = ()  # Slice tuple for tensor indexing


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
            f"tensor={self.tensor is not None}, "
            f"stage={self.stage_id})"
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
            f"tensor={self.tensor is not None}, "
            f"stage={self.stage_id})"
        )
