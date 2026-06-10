"""Run a chunk plan: a windowed pipeline of P2P ops over one communicator.

Chunks arrive in canonical route order on every rank. Ops are batched into
windows of ``window`` routes: a receive lands in its route's window, a relay's
forward in the next one (it needs the data first), a source's send in its own
(the data is local). The same route's send and recv thus share a window number
on both peers — pairing needs no runtime handshake, and the per-rank windows
drift freely, which is what pipelines the chain.

The only true dependency in the system is a relay's send on its own receive,
so only relay receives are awaited at their window; everything else is awaited
once at the end. A plan with no relays therefore runs fully asynchronously.
"""

from collections import defaultdict

import torch
import torch.distributed as dist

from .ir import Chunk


def chunk_comm(chunks: list[Chunk], group: dist.ProcessGroup | None = None, window: int = 16) -> None:
    for chunk in chunks:
        chunk.prepare()

    sends: dict[int, list[Chunk]] = defaultdict(list)
    leaf_recvs: dict[int, list[Chunk]] = defaultdict(list)
    relay_recvs: dict[int, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        win = chunk.route_idx // window
        if chunk.recv_from is not None:
            if chunk.send_to is not None:
                relay_recvs[win].append(chunk)
                sends[win + 1].append(chunk)
            else:
                leaf_recvs[win].append(chunk)
        elif chunk.send_to is not None:
            sends[win].append(chunk)

    pending: list[dist.Work] = []
    for win in sorted(sends.keys() | leaf_recvs.keys() | relay_recvs.keys()):
        ops = [dist.P2POp(dist.isend, c.buffer, c.send_to, group=group) for c in sends[win]]
        ops += [dist.P2POp(dist.irecv, c.buffer, c.recv_from, group=group) for c in leaf_recvs[win]]
        ops += [dist.P2POp(dist.irecv, c.buffer, c.recv_from, group=group) for c in relay_recvs[win]]
        if not ops:
            continue
        works = dist.batch_isend_irecv(ops)
        boundary = len(works) - len(relay_recvs[win])
        pending += works[:boundary]
        for work in works[boundary:]:
            work.wait()
    for work in pending:
        work.wait()

    for chunk in chunks:
        chunk.finalize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
