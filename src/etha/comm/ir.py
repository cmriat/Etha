"""Intermediate Representation for tensor transfer operations."""

import logging
from abc import ABC, abstractmethod
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

    def prepare(self) -> None:
        """Prepare buffer for communication."""
        if self.is_source:
            self._prepare_source()
        else:
            self._prepare_target()

    def _prepare_source(self) -> None:
        """Prepare source bucket buffer."""
        if len(self.entries) == 1:
            chunk = self.entries[0].chunk
            chunk.prepare()
            self.buffer = chunk.buffer
            chunk.buffer = None
            return

        self.buffer = torch.empty(self.total_elems, dtype=self.dtype, device=self.device)

        for entry in self.entries:
            chunk = entry.chunk
            # Manually extract and convert slice (similar to SendChunk.prepare())
            sliced = chunk.tensor[chunk.slice_tuples]
            if hasattr(chunk, "target_dtype") and chunk.target_dtype and chunk.target_dtype != sliced.dtype:
                sliced = sliced.to(chunk.target_dtype)
            flat = sliced.view(-1)
            self.buffer.narrow(0, entry.offset, entry.numel).copy_(flat, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.buffer_ready_event = event

    def _prepare_target(self) -> None:
        """Prepare target bucket buffer."""
        if len(self.entries) == 1:
            chunk = self.entries[0].chunk
            chunk.prepare()
            self.buffer = chunk.buffer
            return

        self.buffer = torch.empty(self.total_elems, dtype=self.dtype, device=self.device)

        for entry in self.entries:
            chunk = entry.chunk
            view = self.buffer.narrow(0, entry.offset, entry.numel).view(chunk.chunk_shape)
            if self.transfer_type == TransferType.SELF_COPY:
                view.copy_(chunk.tensor[chunk.src_slice_tuples], non_blocking=True)
            chunk.buffer = view
        if self.transfer_type == TransferType.SELF_COPY:
            event = torch.cuda.Event()
            event.record()
            self.buffer_ready_event = event

    def launch(self) -> bool:
        """Launch communication operation.

        Returns:
            True if launched, False if still waiting for buffer to be ready.
        """
        if self.buffer_ready_event is not None:
            if not self.buffer_ready_event.query():
                return False
            self.buffer_ready_event = None

        from .transfer_ops import execute_transfer

        self.work = execute_transfer(
            self.buffer,
            self.transfer_type,
            self.is_source,
            self.src_rank,
            self.dst_ranks,
        )
        return True

    def is_complete(self) -> bool:
        """Check if communication is complete.

        Returns:
            True if complete, False otherwise.
        """
        if self.work is None:
            return True
        if self.device is not None and self.device.type == "cpu":  # cpu device do not support is_completed()
            self.work.wait()
            self.work = None
            return True
        return self.work.is_completed()

    def finalize(self) -> None:
        """Finalize communication and cleanup."""
        if self.work is not None:
            self.work.wait()
            self.work = None

        if not self.is_source:
            for entry in self.entries:
                entry.chunk.finalize()

        self.buffer = None


@dataclass(slots=True, kw_only=True)
class SendChunk(BaseChunk):
    """Send chunk for source-side operations.

    Extracts tensor slice, optionally converts dtype, sends via transfer_ops.
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
class RecvChunk(BaseChunk):
    """Receive chunk for target-side operations.

    Allocates buffer, receives data, writes back to tensor.
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
class SelfCopyChunk(BaseChunk):
    """Self-copy chunk (local copy within same rank).

    Performs local tensor copy without network communication.
    """

    dst_idx: tuple
    src_slice_tuples: tuple[slice, ...]

    def prepare(self) -> None:
        """Directly reference source data (no copy needed yet)."""
        self.buffer = self.tensor[self.src_slice_tuples]

    def finalize(self) -> None:
        """Copy data to destination slice and cleanup."""
        self.tensor[self.slice_tuples].copy_(self.buffer, non_blocking=True)
        self.buffer = None
