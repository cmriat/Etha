"""Transfer IR.

A ``Route`` declares dataflow: one source cell to its destination endpoints.
Planning compiles each route into a transfer chain over its members; a
``Chunk`` is one rank's action on that chain — receive from the upstream
neighbor, forward to the downstream one. A relay has both ends; the chain head
only sends, the tail only receives. A relay's upstream neighbor is not the
data's origin, hence ``recv_from``/``send_to`` rather than src/dst naming.
"""

from dataclasses import dataclass

import torch

Cell = tuple[int, ...]


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class Endpoint:
    """``order``: replica holders are sorted so every rank picks the same source."""

    rank: int
    cell: Cell


@dataclass(frozen=True, slots=True, kw_only=True)
class Route:
    src: Endpoint
    dsts: tuple[Endpoint, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class M2MMap:
    routes: list[Route]
    source_num_slicers: list[int]
    target_num_slicers: list[int]


@dataclass(slots=True, kw_only=True)
class Chunk:
    """Not frozen: ``buffer`` is staged in ``prepare`` and dropped in ``finalize``.

    A local self-copy has neither ``recv_from`` nor ``send_to`` and both
    tensors set. A relay forwards from the same buffer it receives into.
    """

    route_idx: int
    weight: str | None = None
    hop: int = 0
    recv_from: int | None = None
    send_to: int | None = None
    src_tensor: torch.Tensor | None = None
    dst_tensor: torch.Tensor | None = None
    src_slice: tuple[slice, ...] = ()
    dst_slice: tuple[slice, ...] = ()
    transfer_dtype: torch.dtype | None = None
    buffer: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.transfer_dtype is None:
            tensor = self.src_tensor if self.src_tensor is not None else self.dst_tensor
            if tensor is not None:
                self.transfer_dtype = tensor.dtype

    def prepare(self) -> None:
        """Stage the wire buffer.

        Reading side: contiguous wire-dtype copy of the source slice (no-op
        when already contiguous and same dtype). Receiving side: land the wire
        op directly in the destination view when layout allows, else a staging
        buffer that ``finalize`` scatters.
        """
        if self.src_tensor is not None:
            buffer = self.src_tensor[self.src_slice]
            if not buffer.is_contiguous() or buffer.dtype != self.transfer_dtype:
                buffer = torch.empty(buffer.shape, dtype=self.transfer_dtype, device=buffer.device).copy_(buffer)
        else:
            view = self.dst_tensor[self.dst_slice]
            if view.is_contiguous() and view.dtype == self.transfer_dtype:
                buffer = view
            else:
                buffer = torch.empty(view.shape, dtype=self.transfer_dtype, device=view.device)
        self.buffer = buffer

    def finalize(self) -> None:
        if self.dst_tensor is not None:
            dst = self.dst_tensor[self.dst_slice]
            if dst.data_ptr() != self.buffer.data_ptr():
                dst.copy_(self.buffer, non_blocking=True)
        self.buffer = None
