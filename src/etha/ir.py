"""Transfer IR.

``Route`` 声明数据流:一个源 cell → 一组目标 endpoint。planning 把每条 route 编成
``Chunk``(每个参与 rank 一个动作):单目标走 P2P(src→dst 直发),多目标走 NCCL
broadcast 子组(root=源,零冗余)——无中继链。
"""

import math
from dataclasses import dataclass
from enum import Enum

import torch

Cell = tuple[int, ...]


class Transport(Enum):
    """过线方式(按 remote dst 数定):1 个 P2P,>1 个 NCCL broadcast 子组(零冗余)。

    dst 落在源 rank 上 = LOCAL 本地自拷(无 wire)。cross-world 源/目标不相交,
    不会有 LOCAL;within-world reshard(如恒等)会有。
    """

    P2P = "p2p"
    BROADCAST = "broadcast"
    LOCAL = "local"


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class Endpoint:
    """``order``: replica holders are sorted so every rank picks the same source."""

    rank: int
    cell: Cell


@dataclass(frozen=True, slots=True, kw_only=True)
class Route:
    src: Endpoint
    dsts: tuple[Endpoint, ...]
    kind: Transport = Transport.P2P


@dataclass(frozen=True, slots=True, kw_only=True)
class M2MMap:
    routes: list[Route]
    source_num_slicers: list[int]
    target_num_slicers: list[int]


@dataclass(slots=True, kw_only=True)
class Chunk:
    """一个 rank 在一条 route 上的动作。Not frozen:``buffer`` 在 prepare 暂存、finalize 丢弃。

    角色由谁在场推出(不存 is_source/is_target):src_tensor 在场 = 本 rank 是源
    (读 src_slice → buffer);dst_tensor 在场 = 本 rank 是目标(buffer → 写 dst_slice)。
    transport=P2P:对端 = dst_ranks[0](源侧)/ src_rank(收侧);
    transport=BROADCAST:子组 {src_rank}∪dst_ranks,root=src_rank。
    """

    route_idx: int
    transport: Transport
    weight: str | None = None
    src_rank: int
    dst_ranks: tuple[int, ...] = ()
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

    @property
    def view_shape(self) -> torch.Size:  # 用真实 tensor[slice].shape(EP local clamp 后),不能拿 slice 名义 stop-start(全局尺寸)
        if self.src_tensor is not None:
            return self.src_tensor[self.src_slice].shape
        return self.dst_tensor[self.dst_slice].shape

    @property
    def nbytes(self) -> int:
        return math.prod(self.view_shape) * self.transfer_dtype.itemsize


@dataclass(slots=True, kw_only=True)
class Bucket:
    """同 (src_rank, dst_ranks, transport, 层) 的 chunk 聚成一个 wire op。

    ``prepare`` 拼一个连续大 buffer(各 chunk = offset view,源端把 src_slice copy
    进来),``launch`` 跑一次 broadcast/P2P,``finalize`` 把收到的切回各 dst_tensor。
    op 数从 per-chunk → per-(子组,层),消掉 per-chunk launch 间隙(通信 gap)。
    experts 限层不跨(撞 vLLM layerwise reload 逐层 process 上界),replicate 可跨层。
    """

    chunks: list[Chunk]
    transport: Transport
    src_rank: int
    dst_ranks: tuple[int, ...]
    buffer: torch.Tensor | None = None

    def prepare(self) -> None:
        off, total = [], 0
        for c in self.chunks:
            off.append(total)
            total += c.nbytes
        device = next(t.device for c in self.chunks for t in (c.src_tensor, c.dst_tensor) if t is not None)
        self.buffer = torch.empty(total, dtype=torch.uint8, device=device)
        for c, o in zip(self.chunks, off):
            c.buffer = self.buffer[o : o + c.nbytes].view(c.transfer_dtype).view(c.view_shape)
            if c.src_tensor is not None:
                c.buffer.copy_(c.src_tensor[c.src_slice])

    def finalize(self) -> None:
        for c in self.chunks:
            if c.dst_tensor is not None:
                c.dst_tensor[c.dst_slice].copy_(c.buffer, non_blocking=True)
            c.buffer = None
        self.buffer = None
