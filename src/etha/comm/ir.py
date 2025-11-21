"""Intermediate Representation for tensor transfer operations."""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass

import torch
import torch.distributed as dist

from . import utils

logger = logging.getLogger(__name__)


class TransferType(Enum):
    """Transfer operation types."""

    SELF_COPY = "self_copy"  # Local copy within same rank
    P2P = "p2p"  # Point-to-point transfer between two ranks
    BROADCAST = "broadcast"  # One-to-many transfer


@dataclass(slots=True, kw_only=True)
class BaseChunk(ABC):
    """Abstract base class for transfer chunks."""

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

    # Source info
    src_rank: int  # Rank that owns this data
    src_idx: tuple  # Multi-dimensional index in source tensor

    # Destination info
    dst_ranks: tuple[int, ...]  # Target ranks (len > 1 triggers broadcast)

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"type={self.transfer_type.name[:3] if self.transfer_type else '???'}, "
            f"src={self.src_rank}→{self.dst_ranks}, "
            f"tensor={self.tensor is not None})"
        )

    @abstractmethod
    def prepare(self) -> None:
        """Prepare buffer from tensor slice.

        Extracts data from self.tensor into self.buffer, ready for communication.
        Subclasses implement specific slicing and dtype conversion logic.
        """

    @abstractmethod
    def launch(self) -> None:
        """Launch communication operation.

        Initiates async communication (isend/irecv/broadcast) or performs
        synchronous local copy. Sets self.work for async operations.
        """

    @abstractmethod
    def finalize(self) -> None:
        """Finalize communication and cleanup.

        Waits for async work to complete, writes data to target tensor if needed,
        and cleans up buffers.
        """


@dataclass(slots=True, kw_only=True)
class BucketEntry:
    """Bucket offset entry."""

    offset: int
    numel: int
    chunk: BaseChunk


@dataclass(slots=True, kw_only=True)
class Bucket:
    """Bucket for transfer operations."""

    transfer_type: TransferType
    is_source: bool
    dst_ranks: tuple[int, ...] | None = None
    src_rank: int | None = None
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    buffer: torch.Tensor | None = None
    work: "torch.distributed.Work | None" = None
    buffer_ready_event: torch.cuda.Event | None = None
    total_elems: int
    key: tuple
    entries: list[BucketEntry]

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        kind = "src" if self.is_source else "dst"
        return f"Bucket({kind}, key={self.key} entries_len={len(self.entries)} src_rank={self.src_rank}→dst_ranks={self.dst_ranks})"


@dataclass(slots=True, kw_only=True)
class SendChunk(BaseChunk, ABC):
    """Abstract base for send chunks (source-side operations).

    Common behavior: extract tensor slice, optionally convert dtype, cleanup after send.
    Subclasses only implement launch() with specific communication primitive.
    """

    target_dtype: torch.dtype | None = None

    def prepare(self) -> None:
        """Extract and optionally convert tensor slice for sending."""
        self.buffer = self.tensor[self.slice_tuples].contiguous()
        if self.target_dtype and self.target_dtype != self.buffer.dtype:
            self.buffer = self.buffer.to(self.target_dtype)

    def finalize(self) -> None:
        """Wait for send completion and cleanup."""
        if self.work is not None:
            self.work.wait()
            self.work = None
        self.buffer = None


@dataclass(slots=True, kw_only=True)
class RecvChunk(BaseChunk, ABC):
    """Abstract base for receive chunks (target-side operations).

    Common behavior: allocate buffer, write back to tensor, cleanup after receive.
    Subclasses only implement launch() with specific communication primitive.
    """

    dst_idx: tuple

    def prepare(self) -> None:
        """Allocate buffer for receiving data."""
        self.buffer = self.tensor[self.slice_tuples].contiguous()

    def finalize(self) -> None:
        """Wait for receive completion, write to tensor, and cleanup."""
        if self.work is not None:
            self.work.wait()
            self.work = None
        self.tensor[self.slice_tuples].copy_(self.buffer, non_blocking=True)
        self.buffer = None


@dataclass(slots=True, kw_only=True)
class SendP2PChunk(SendChunk):
    """Point-to-point send chunk.

    Sends data to a single destination rank via dist.isend().
    """

    def launch(self) -> None:
        """Launch async P2P send."""
        self.work = dist.isend(self.buffer, dst=self.dst_ranks[0])


@dataclass(slots=True, kw_only=True)
class SendBroadcastChunk(SendChunk):
    """Broadcast send chunk.

    Sends data to multiple destination ranks via dist.broadcast().
    """

    def launch(self) -> None:
        """Launch async broadcast send."""
        group_ranks = sorted([self.src_rank, *self.dst_ranks])
        group = utils.get_or_create_process_group(group_ranks)
        self.work = dist.broadcast(self.buffer, src=self.src_rank, group=group, async_op=True)


@dataclass(slots=True, kw_only=True)
class RecvP2PChunk(RecvChunk):
    """Point-to-point receive chunk.

    Receives data from a single source rank via dist.irecv().
    """

    def launch(self) -> None:
        """Launch async P2P receive."""
        self.work = dist.irecv(self.buffer, src=self.src_rank)


@dataclass(slots=True, kw_only=True)
class RecvBroadcastChunk(RecvChunk):
    """Broadcast receive chunk.

    Receives data from source rank via dist.broadcast().
    """

    def launch(self) -> None:
        """Launch async broadcast receive."""
        group_ranks = sorted([self.src_rank, *self.dst_ranks])
        group = utils.get_or_create_process_group(group_ranks)
        self.work = dist.broadcast(self.buffer, src=self.src_rank, group=group, async_op=True)


@dataclass(slots=True, kw_only=True)
class SelfCopyChunk(BaseChunk):
    """Self-copy chunk (local copy within same rank).

    Performs local tensor copy without network communication.
    """

    dst_idx: tuple
    src_slice_tuples: tuple[slice, ...]

    def prepare(self) -> None:
        """Directly reference source data (no copy needed yet)."""
        self.buffer = self.tensor[self.src_slice_tuples]

    def launch(self) -> None:
        """No async work for local copy."""
        self.work = None

    def finalize(self) -> None:
        """Copy data to destination slice and cleanup."""
        self.tensor[self.slice_tuples].copy_(self.buffer, non_blocking=True)
        self.buffer = None
