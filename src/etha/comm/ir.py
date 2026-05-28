"""Intermediate Representation for tensor transfer operations."""

import logging
from dataclasses import field, dataclass

import torch
import msgspec
import torch.distributed as dist

from .transfer import Transferable, TransferType

logger = logging.getLogger(__name__)


class Endpoint(msgspec.Struct, frozen=True):
    """One chunk location: a rank and its cell coordinate in the transfer grid.

    ``cell`` is the multi-dimensional index of the chunk within that rank's
    local shard (fed to ``get_slice_from_multi_index`` to get the byte slice).
    Leaf data (int + tuple of ints), so no reference cycles.
    """

    rank: int
    cell: tuple[int, ...]


class Route(msgspec.Struct, frozen=True):
    """One source cell's delivery plan: ``src`` endpoint to a set of ``dsts``.

    ``kind`` is the route-level classification, set once at construction:
    SHADOW (empty ``dsts``, reduce-only), BROADCAST (>1 dst), or P2P (1 dst).
    SELF_COPY is not a route kind — it is refined per-dst at execution when a
    dst lands on the source rank.
    """

    src: Endpoint
    dsts: tuple[Endpoint, ...]
    kind: TransferType


_REDUCE_OP_MAP = {
    "sum": dist.ReduceOp.SUM,
    "avg": dist.ReduceOp.AVG,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
    "product": dist.ReduceOp.PRODUCT,
}


@dataclass(slots=True, kw_only=True)
class Chunk(Transferable):
    """Unified chunk for all transfer operations."""

    chunk_shape: tuple[int, ...]  # Shape of the data being transferred
    tensor: torch.Tensor | None = None
    slice_tuples: tuple[slice, ...] = ()  # Slice tuple for tensor indexing (dst for recv/self-copy)
    transfer_dtype: torch.dtype | None = None  # Wire dtype (None = use tensor.dtype, set in __post_init__)
    src_idx: tuple  # Multi-dimensional index in source tensor
    dst_idx: tuple | None = None  # Multi-dimensional index in target tensor (recv/self-copy only)
    src_slice_tuples: tuple[slice, ...] | None = None  # Source slice (self-copy only)
    # NCCL sub-groups + reduce_op for collapsing source Partial to Replicate
    # before send. Routing emits a chunk per (member, cell) so every member
    # reaches the collective; SHADOW chunks participate without shipping.
    source_partial_groups: list[tuple[dist.ProcessGroup, str]] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.transfer_dtype is None:
            self.transfer_dtype = self.tensor.dtype

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        return (
            f"Chunk("
            f"type={self.transfer_type.name[:3] if self.transfer_type else '???'}, "
            f"src={self.src_rank}→{self.dst_ranks}, "
            f"tensor={self.tensor is not None})"
        )

    def prepare(self, contiguous: bool = True) -> None:
        """Prepare source/target buffer.

        Source side performs (in order): slice → in-place all-reduce on
        Partial sub-groups (in source dtype) → cast to ``transfer_dtype``.
        Reducing before the cast matches DTensor ``Partial → Replicate``
        semantics; running the all-reduce in the (possibly lower-precision)
        wire dtype would change numerical results.
        """
        if self.transfer_type == TransferType.SELF_COPY:
            buffer = self.tensor[self.src_slice_tuples]
        else:
            buffer = self.tensor[self.slice_tuples]
            if contiguous:
                buffer = buffer.contiguous()

        if self.is_source and self.transfer_type != TransferType.SELF_COPY:
            if self.source_partial_groups:
                # all_reduce is in-place; ensure we own the storage so the
                # source tensor isn't mutated. Storage-level alias check —
                # ``tensor.data_ptr()`` accounts for ``storage_offset`` and
                # would miss non-zero-offset slices that still alias.
                if buffer.untyped_storage().data_ptr() == self.tensor.untyped_storage().data_ptr():
                    buffer = buffer.contiguous().clone()
                for group, op_str in self.source_partial_groups:
                    dist.all_reduce(buffer, op=_REDUCE_OP_MAP[op_str], group=group)
            if self.transfer_dtype and self.transfer_dtype != buffer.dtype:
                buffer = buffer.to(self.transfer_dtype)

        self.buffer = buffer

    def finalize(self) -> None:
        """Finalize communication and cleanup."""
        if self.work is not None:
            self.work.wait()
            self.work = None
        if self.transfer_type == TransferType.SELF_COPY or not self.is_source:
            buffer = self.buffer
            # Target: convert from transfer_dtype to tensor.dtype if different
            if buffer.dtype != self.tensor.dtype:
                buffer = buffer.to(self.tensor.dtype)
            self.tensor[self.slice_tuples].copy_(buffer, non_blocking=True)
        self.buffer = None

    @property
    def bucket_key(self) -> tuple:
        """Return bucket grouping key.

        ``cell_key`` is added for SHADOW Partial chunks only — they all share
        ``dst_ranks=()`` and would otherwise bundle across cells, making the
        bucket's all_reduce sequence per-rank-specific and out of sync with
        peer ranks whose matching cells live in separate buckets. PRIMARY
        chunks already differ by ``dst_ranks`` across cells.
        """
        partial_sig: tuple = ()
        cell_key: tuple = ()
        if self.source_partial_groups:
            partial_sig = tuple((id(g), op) for g, op in self.source_partial_groups)
            if not self.dst_ranks:
                cell_key = self.src_idx
        return (self.src_rank, self.dst_ranks, partial_sig, cell_key)


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
        """Prepare buffer for communication.

        Source-side Partial reduce and dtype cast both live inside
        ``Chunk.prepare``; this method only assembles entries into the
        bucket buffer.
        """
        if len(self.entries) == 1:
            chunk = self.entries[0].chunk
            chunk.prepare()
            self.buffer = chunk.buffer.view(torch.uint8)
            if self.is_source:
                chunk.buffer = None
            return

        self.buffer = torch.empty(self.total_bytes, dtype=torch.uint8, device=self.device)
        needs_copy = self.is_source or self.transfer_type == TransferType.SELF_COPY

        for entry in self.entries:
            chunk = entry.chunk
            dtype = chunk.transfer_dtype
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
