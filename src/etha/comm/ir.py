"""Intermediate Representation for tensor transfer operations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .transfer_ops import Transferable, TransferType

logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class BaseChunk(Transferable, ABC):
    """Abstract base class for transfer chunks."""

    chunk_shape: tuple[int, ...]  # Shape of the data being transferred
    tensor: torch.Tensor | None = None
    slice_tuples: tuple[slice, ...] = ()  # Slice tuple for tensor indexing
    target_dtype: torch.dtype | None = None  # Dtype conversion (None means no conversion)
    src_idx: tuple  # Multi-dimensional index in source tensor

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

    @property
    def bucket_key(self) -> tuple:
        """Return bucket grouping key (src_rank, dst_ranks)."""
        return (self.src_rank, self.dst_ranks)


@dataclass(slots=True, kw_only=True)
class BucketEntry:
    """Bucket offset entry (byte-based)."""

    offset: int  # Byte offset in bucket buffer
    nbytes: int  # Size in bytes
    chunk: BaseChunk


@dataclass(slots=True, kw_only=True)
class Bucket(Transferable):
    """Bucket for transfer operations (byte-based buffer)."""

    total_bytes: int
    key: tuple
    entries: list[BucketEntry]
    device: torch.device | None = None
    buffer_ready_event: torch.cuda.Event | None = None

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
        """Prepare source bucket buffer (byte-based)."""
        if len(self.entries) == 1:
            chunk = self.entries[0].chunk
            chunk.prepare()
            self.buffer = chunk.buffer.view(torch.uint8)
            chunk.buffer = None
            return

        self.buffer = torch.empty(self.total_bytes, dtype=torch.uint8, device=self.device)

        for entry in self.entries:
            chunk = entry.chunk
            # Manually extract and convert slice (similar to SendChunk.prepare())
            sliced = chunk.tensor[chunk.slice_tuples]
            if hasattr(chunk, "target_dtype") and chunk.target_dtype and chunk.target_dtype != sliced.dtype:
                sliced = sliced.to(chunk.target_dtype)
            flat_bytes = sliced.view(-1).view(torch.uint8)
            self.buffer.narrow(0, entry.offset, entry.nbytes).copy_(flat_bytes, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.buffer_ready_event = event

    def _prepare_target(self) -> None:
        """Prepare target bucket buffer (byte-based)."""
        if len(self.entries) == 1:
            chunk = self.entries[0].chunk
            chunk.prepare()
            self.buffer = chunk.buffer.view(torch.uint8)
            return

        self.buffer = torch.empty(self.total_bytes, dtype=torch.uint8, device=self.device)

        for entry in self.entries:
            chunk = entry.chunk
            # Determine dtype for this chunk
            dtype = chunk.tensor.dtype
            numel = entry.nbytes // dtype.itemsize
            view = self.buffer.narrow(0, entry.offset, entry.nbytes).view(dtype)[:numel].view(chunk.chunk_shape)
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

        self.work = self.execute()
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
    """Send chunk for source-side operations."""

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
    """Receive chunk for target-side operations."""

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
