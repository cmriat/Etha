"""Run a chunk plan: a windowed pipeline of P2P ops over one communicator.

P2P pairing is FIFO per peer pair (no tags on NCCL), so both ends must post
every message in the same global order. That order is (window, route):

    chain [src, d1, d2]:   src --edge 0--> d1 --edge 1--> d2
    edge i of a route lives in window  route_idx // window + i

Both ends of an edge derive the same window from ``Chunk.hop`` (the sender's
chain position), and within a window every rank appends ops in route order.
A relay receives in one window and forwards in the next, which is exactly its
data dependency; the wait at each window boundary enforces it. Ranks drift
through windows independently — that drift is what pipelines the chain.

Memory is streamed through the same loop: a chunk's wire buffer is staged
(``prepare``) right before its first window and dropped after its last one,
and destination buffers can be allocated on demand (``target_alloc``) and
handed off per finished weight (``on_complete``) — peak memory is what is in
flight, not the model.
"""

from collections import defaultdict
from collections.abc import Callable

import torch
import torch.distributed as dist

from .ir import Chunk


def chunk_comm(
    chunks: list[Chunk],
    group: dist.ProcessGroup | None = None,
    window: int = 16,
    target_alloc: Callable[[str], torch.Tensor] | None = None,
    on_complete: Callable[[str, torch.Tensor], None] | None = None,
) -> None:
    sends: dict[int, list[Chunk]] = defaultdict(list)
    recvs: dict[int, list[Chunk]] = defaultdict(list)
    finish: dict[int, list[Chunk]] = defaultdict(list)
    pending: dict[str, int] = defaultdict(int)
    allocated: dict[str, torch.Tensor] = {}

    def is_dst(chunk: Chunk) -> bool:
        return len(chunk.dst_slice) > 0

    def bind(chunk: Chunk) -> None:
        if chunk.dst_tensor is None and is_dst(chunk) and target_alloc is not None:
            if chunk.weight not in allocated:
                allocated[chunk.weight] = target_alloc(chunk.weight)
            chunk.dst_tensor = allocated[chunk.weight]
        if chunk.buffer is None:
            chunk.prepare()

    def done(chunk: Chunk) -> None:
        if is_dst(chunk):
            chunk.finalize()
            if chunk.weight is not None:
                pending[chunk.weight] -= 1
                if pending[chunk.weight] == 0 and on_complete is not None:
                    on_complete(chunk.weight, allocated.pop(chunk.weight, chunk.dst_tensor))
        else:
            chunk.buffer = None

    local: list[Chunk] = []
    for chunk in chunks:
        if is_dst(chunk) and chunk.weight is not None:
            pending[chunk.weight] += 1
        base = chunk.route_idx // window
        last = None
        if chunk.recv_from is not None:
            recvs[base + chunk.hop - 1].append(chunk)
            last = base + chunk.hop - 1
        if chunk.send_to is not None:
            sends[base + chunk.hop].append(chunk)
            last = base + chunk.hop
        if last is None:
            local.append(chunk)
        else:
            finish[last].append(chunk)

    for chunk in local:
        bind(chunk)
        done(chunk)

    for win in sorted(sends.keys() | recvs.keys()):
        for chunk in sends[win]:
            bind(chunk)
        for chunk in recvs[win]:
            bind(chunk)
        ops = [dist.P2POp(dist.isend, c.buffer, c.send_to, group=group) for c in sends[win]]
        ops += [dist.P2POp(dist.irecv, c.buffer, c.recv_from, group=group) for c in recvs[win]]
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        for chunk in finish[win]:
            done(chunk)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
