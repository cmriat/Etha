"""Intermediate Representation for tensor transfer operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from .transfer import Transferable, TransferType

logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class Chunk(Transferable):
    """Unified chunk for all transfer operations."""

    chunk_shape: tuple[int, ...]  # Shape of the data being transferred
    tensor: torch.Tensor | None = None
    slice_tuples: tuple[slice, ...] = ()  # Slice tuple for tensor indexing (dst for recv/self-copy)
    target_dtype: torch.dtype | None = None  # Dtype conversion (None means no conversion)
    src_idx: tuple  # Multi-dimensional index in source tensor
    dst_idx: tuple | None = None  # Multi-dimensional index in target tensor (recv/self-copy only)
    src_slice_tuples: tuple[slice, ...] | None = None  # Source slice (self-copy only)

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        return (
            f"Chunk("
            f"type={self.transfer_type.name[:3] if self.transfer_type else '???'}, "
            f"src={self.src_rank}→{self.dst_ranks}, "
            f"tensor={self.tensor is not None})"
        )

    def prepare(self, contiguous: bool = True) -> None:
        """Prepare buffer from tensor slice.

        Args:
            contiguous: Whether to make buffer contiguous (default True)
        """
        if self.transfer_type == TransferType.SELF_COPY:
            buffer = self.tensor[self.src_slice_tuples]
        else:
            buffer = self.tensor[self.slice_tuples]
            if contiguous:
                buffer = buffer.contiguous()
        if self.is_source and self.target_dtype and self.target_dtype != buffer.dtype:
            buffer = buffer.to(self.target_dtype)
        self.buffer = buffer

    def finalize(self) -> None:
        """Finalize communication and cleanup."""
        if self.work is not None:
            self.work.wait()
            self.work = None
        if self.transfer_type == TransferType.SELF_COPY or not self.is_source:
            self.tensor[self.slice_tuples].copy_(self.buffer, non_blocking=True)
        self.buffer = None

    @property
    def bucket_key(self) -> tuple:
        """Return bucket grouping key (src_rank, dst_ranks)."""
        return (self.src_rank, self.dst_ranks)


@dataclass(slots=True, kw_only=True)
class BucketEntry:
    """Bucket offset entry (byte-based)."""

    offset: int  # Byte offset in bucket buffer
    nbytes: int  # Size in bytes
    chunk: Chunk


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
        # Single entry fast path
        if len(self.entries) == 1:
            chunk = self.entries[0].chunk
            chunk.prepare()
            self.buffer = chunk.buffer.view(torch.uint8)
            if self.is_source:
                chunk.buffer = None
            return

        # Multi-entry: allocate bucket buffer
        self.buffer = torch.empty(self.total_bytes, dtype=torch.uint8, device=self.device)
        needs_copy = self.is_source or self.transfer_type == TransferType.SELF_COPY

        for entry in self.entries:
            chunk = entry.chunk
            dtype = chunk.tensor.dtype
            numel = entry.nbytes // dtype.itemsize
            buffer_slice = self.buffer.narrow(0, entry.offset, entry.nbytes).view(dtype)[:numel].view(chunk.chunk_shape)

            if needs_copy:
                chunk.prepare(contiguous=False)
                buffer_slice.copy_(chunk.buffer, non_blocking=True)
                chunk.buffer = None
            else:
                chunk.buffer = buffer_slice

        if needs_copy:
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
