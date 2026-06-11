"""Build the one cross-world communicator from a TCP store.

torch has no public path to a functional-API-usable ProcessGroup spanning two
already-initialized worlds: directly constructed backends are absent from the
group registry that ``P2POp`` consults, and ``new_group`` only carves subsets
of the default group. So this uses two stable internals —
``_new_process_group_helper`` (the body of ``new_group``) plus a registry
entry mapping cross-world ranks to themselves.

Caller's process must already have a default group (engine workers do).
``rank`` is the cross-world numbering shared with the planner's mesh tensors.
"""

import datetime

import torch.distributed as dist
from torch.distributed.distributed_c10d import _world, _new_process_group_helper


def create_cross_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    backend: str = "nccl",
    timeout_s: float = 300.0,
) -> dist.ProcessGroup:
    timeout = datetime.timedelta(seconds=timeout_s)
    store = dist.TCPStore(host, port, world_size, is_master=rank == 0, timeout=timeout)
    pg, _ = _new_process_group_helper(
        group_size=world_size,
        group_rank=rank,
        global_ranks_in_group=list(range(world_size)),
        backend=backend,
        store=dist.PrefixStore("etha", store),
        group_name="etha_cross",
        timeout=timeout,
    )
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg
