"""Intermediate Representation for tensor transfer operations."""

import math
import logging
from dataclasses import field, dataclass

import torch
import msgspec
import torch.distributed as dist

from .transfer import Transport, _execute_p2p, _execute_broadcast

logger = logging.getLogger(__name__)


class Endpoint(msgspec.Struct, frozen=True):
    """A chunk location: a rank and its cell (multi-index) in the transfer grid."""

    rank: int
    cell: tuple[int, ...]


class Route(msgspec.Struct, frozen=True):
    """One source cell's delivery: ``src`` endpoint to a set of dst endpoints.

    ``kind`` is fixed at construction: empty ``dsts`` -> NONE (reduce-only),
    else BROADCAST (>1) or P2P (1). LOCAL is not a route kind; a dst that lands
    on the source rank is refined to a local copy when chunks are built.
    """

    src: Endpoint
    dsts: tuple[Endpoint, ...]
    kind: Transport


class M2MMap(msgspec.Struct):
    """Mesh-to-mesh topology (shape-independent, reusable across batches).

    ``routes`` is a flat list of per-cell delivery plans (see ``Route``);
    ``source_num_slicers``/``target_num_slicers`` describe how each side
    partitions the tensor; ``source_partial_reductions`` lists
    ``(mesh_dim, reduce_op)`` per source Partial dim (empty when none).
    """

    routes: list[Route] | None = None
    source_num_slicers: list[int] | None = None
    target_num_slicers: list[int] | None = None
    source_partial_reductions: list[tuple[int, str]] = []


_REDUCE_OP_MAP = {
    "sum": dist.ReduceOp.SUM,
    "avg": dist.ReduceOp.AVG,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
    "product": dist.ReduceOp.PRODUCT,
}


@dataclass(slots=True, kw_only=True)
class Chunk:
    """A shape-dependent transfer descriptor: a tensor region plus its role.

    Not a transfer unit — a ``Bucket`` runs the wire op; a chunk only describes
    one tensor region and how to prepare/finalize it.
    """

    transport: Transport
    is_source: bool
    is_target: bool
    src_rank: int
    dst_ranks: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    tensor: torch.Tensor | None = None
    src_slice: tuple[slice, ...] = ()  # read here when is_source
    dst_slice: tuple[slice, ...] = ()  # written here when is_target
    transfer_dtype: torch.dtype | None = None  # Wire dtype (None = use tensor.dtype, set in __post_init__)
    src_idx: tuple  # Multi-dimensional index in source tensor
    dst_idx: tuple | None = None  # Multi-dimensional index in target tensor (consumers only)
    # NCCL sub-groups + reduce_op for collapsing source Partial to Replicate
    # before send. Routing emits a chunk per (member, cell) so every member
    # reaches the collective; reduce-only (NONE transport) chunks participate
    # without shipping.
    source_partial_groups: list[tuple[dist.ProcessGroup, str]] | None = field(default=None)
    # Set for every chunk of a Partial transfer (both send and recv ends, keyed off
    # the M2MMap so it's symmetric). Partial chunks must not coalesce: their per-cell
    # all_reduce must stay aligned across the reduce group, and a sender-only signal
    # would desync send/recv bucketing. See chunk_to_bucket_ops.
    no_coalesce: bool = False
    buffer: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.transfer_dtype is None:
            self.transfer_dtype = self.tensor.dtype

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        role = "".join(c for c, on in (("S", self.is_source), ("T", self.is_target)) if on) or "-"
        return (
            f"Chunk("
            f"{self.transport.name[:3]}/{role}, "
            f"src={self.src_rank}→{self.dst_ranks}, "
            f"tensor={self.tensor is not None})"
        )

    def prepare(self, contiguous: bool = True) -> None:
        """Prepare the buffer.

        ``is_source`` reads ``src_slice`` then performs (in order): in-place
        all-reduce on Partial sub-groups (in source dtype) → cast to
        ``transfer_dtype``. Reducing before the cast matches DTensor
        ``Partial → Replicate`` semantics; running the all-reduce in the
        (possibly lower-precision) wire dtype would change numerical results.
        A consume-only chunk (recv) instead views ``dst_slice`` so the wire
        op lands directly in the target.
        """
        if self.is_source:
            buffer = self.tensor[self.src_slice]
            if contiguous:
                buffer = buffer.contiguous()
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
        else:
            buffer = self.tensor[self.dst_slice]
            if contiguous:
                buffer = buffer.contiguous()

        self.buffer = buffer

    def finalize(self) -> None:
        """Write the produced/received buffer back to the target tensor."""
        if self.is_target:
            buffer = self.buffer
            if buffer.dtype != self.tensor.dtype:
                buffer = buffer.to(self.tensor.dtype)
            self.tensor[self.dst_slice].copy_(buffer, non_blocking=True)
        self.buffer = None

    @property
    def bucket_key(self) -> tuple:
        """Return bucket grouping key.

        ``transport`` is in the key so a local (self-copy) chunk never bundles
        with a co-located broadcast source chunk: they share ``src_rank`` and
        ``dst_ranks`` but must run different ops and produce buffers of
        different sizes across the broadcast group.

        ``cell_key`` is added for reduce-only Partial chunks only — they all
        share ``dst_ranks=()`` and would otherwise bundle across cells, making
        the bucket's all_reduce sequence per-rank-specific and out of sync with
        peer ranks whose matching cells live in separate buckets. Shipping
        chunks already differ by ``dst_ranks`` across cells.
        """
        partial_sig: tuple = ()
        cell_key: tuple = ()
        if self.source_partial_groups:
            partial_sig = tuple((id(g), op) for g, op in self.source_partial_groups)
            if not self.dst_ranks:
                cell_key = self.src_idx
        return (self.src_rank, self.dst_ranks, partial_sig, cell_key, self.transport)

    @property
    def nbytes(self) -> int:
        """Byte size on the wire (transfer dtype)."""
        return math.prod(self.chunk_shape) * self.transfer_dtype.itemsize


@dataclass(slots=True, kw_only=True)
class Bucket:
    """A transfer unit: one buffer + one wire op for its chunks.

    Identity (transport/role/ranks/key) is uniform across a bucket's chunks
    (they share a ``bucket_key``), so it is read from the first chunk. Byte
    offsets (the prefix sum of ``chunk.nbytes``) are computed once at
    construction since a bucket is reused across transfers.
    """

    chunks: list[Chunk]
    device: torch.device | None = None
    buffer_ready_event: torch.cuda.Event | None = None
    buffer: torch.Tensor | None = None
    work: dist.Work | None = None
    offsets: list[int] = field(init=False)

    def __post_init__(self) -> None:
        offsets, cursor = [], 0
        for chunk in self.chunks:
            offsets.append(cursor)
            cursor += chunk.nbytes
        self.offsets = offsets

    @property
    def transport(self) -> Transport:
        return self.chunks[0].transport

    @property
    def is_source(self) -> bool:
        return self.chunks[0].is_source

    @property
    def is_target(self) -> bool:
        return self.chunks[0].is_target

    @property
    def src_rank(self) -> int:
        return self.chunks[0].src_rank

    @property
    def dst_ranks(self) -> tuple[int, ...]:
        return self.chunks[0].dst_ranks

    @property
    def key(self) -> tuple:
        return self.chunks[0].bucket_key

    @property
    def total_bytes(self) -> int:
        return self.offsets[-1] + self.chunks[-1].nbytes

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        role = "".join(c for c, on in (("S", self.is_source), ("T", self.is_target)) if on) or "-"
        return f"Bucket({self.transport.name[:3]}/{role}, key={self.key} n={len(self.chunks)} src_rank={self.src_rank}→dst_ranks={self.dst_ranks})"

    def prepare(self) -> None:
        """Assemble the bucket buffer from its chunks.

        Source-side Partial reduce and dtype cast both live inside
        ``Chunk.prepare``. A producing chunk's data is copied into the bucket;
        the per-chunk buffer is kept only when the chunk also ``is_target``
        (self-copy), so ``finalize`` can write it to the target. A consume-only
        recv instead points its buffer at the bucket slice to land directly.
        """
        if len(self.chunks) == 1:
            chunk = self.chunks[0]
            chunk.prepare()
            self.buffer = chunk.buffer.view(torch.uint8)
            if not chunk.is_target:
                chunk.buffer = None
            return

        self.buffer = torch.empty(self.total_bytes, dtype=torch.uint8, device=self.device)

        for offset, chunk in zip(self.offsets, self.chunks, strict=True):
            dtype = chunk.transfer_dtype
            nbytes = chunk.nbytes
            numel = nbytes // dtype.itemsize
            buffer_slice = self.buffer.narrow(0, offset, nbytes).view(dtype)[:numel].view(chunk.chunk_shape)

            if self.is_source:
                chunk.prepare(contiguous=False)
                buffer_slice.copy_(chunk.buffer, non_blocking=True)
                chunk.buffer = buffer_slice if chunk.is_target else None
            else:
                chunk.buffer = buffer_slice

        if self.is_source:
            event = torch.cuda.Event()
            event.record()
            self.buffer_ready_event = event

    def launch(self) -> bool:
        """Issue the wire op once the assembled buffer is ready.

        Returns False if the buffer-assembly event hasn't fired yet; otherwise
        issues the transport (no-op for LOCAL/NONE) and returns True.
        """
        if self.buffer_ready_event is not None:
            if not self.buffer_ready_event.query():
                return False
            self.buffer_ready_event = None

        match self.transport:
            case Transport.LOCAL | Transport.NONE:
                self.work = None
            case Transport.P2P:
                self.work = _execute_p2p(self.buffer, self.is_source, self.src_rank, self.dst_ranks[0])
            case Transport.BROADCAST:
                self.work = _execute_broadcast(self.buffer, self.src_rank, self.dst_ranks)
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

        if self.is_target:
            for chunk in self.chunks:
                chunk.finalize()

        self.buffer = None
