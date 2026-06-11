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
"""

from collections import defaultdict

import torch
import torch.distributed as dist

from .ir import Chunk


def chunk_comm(chunks: list[Chunk], group: dist.ProcessGroup | None = None, window: int = 16) -> None:
    for chunk in chunks:
        chunk.prepare()

    sends: dict[int, list[Chunk]] = defaultdict(list)
    recvs: dict[int, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        base = chunk.route_idx // window
        if chunk.recv_from is not None:
            recvs[base + chunk.hop - 1].append(chunk)
        if chunk.send_to is not None:
            sends[base + chunk.hop].append(chunk)

    for win in sorted(sends.keys() | recvs.keys()):
        ops = [dist.P2POp(dist.isend, c.buffer, c.send_to, group=group) for c in sends[win]]
        ops += [dist.P2POp(dist.irecv, c.buffer, c.recv_from, group=group) for c in recvs[win]]
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    for chunk in chunks:
        chunk.finalize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
