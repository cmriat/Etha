"""Build the cross-world communicator (+ broadcast subgroups) from a TCP store.

torch has no public path to a functional-API-usable ProcessGroup spanning two
already-initialized worlds: directly constructed backends are absent from the
group registry that ``P2POp``/collectives consult, and ``new_group`` only carves
subsets of the default group. So this uses two stable internals —
``_new_process_group_helper`` (the body of ``new_group``) plus a registry entry
mapping cross-world ranks. The same trick builds per-broadcast subgroups: each
``{src}∪dsts`` set gets its own PG over the same TCP store (unique PrefixStore),
and only its members call in — unlike ``new_group``, no all-default-group barrier.

Caller's process must already have a default group (engine workers do).
``rank`` is the cross-world numbering shared with the planner's mesh tensors.
"""

import datetime
import os

import torch.distributed as dist
from torch.distributed.distributed_c10d import _world, _new_process_group_helper

_TIMEOUT_S = float(os.environ.get("ETHA_PG_TIMEOUT", 1800))


def create_cross_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    backend: str = "nccl",
    timeout_s: float = _TIMEOUT_S,
) -> tuple[dist.ProcessGroup, dist.TCPStore]:
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
    return pg, store


def create_broadcast_subgroups(
    store: dist.TCPStore,
    rank: int,
    subgroup_ranks: list[tuple[int, ...]],
    backend: str = "nccl",
    timeout_s: float = _TIMEOUT_S,
) -> dict[tuple[int, ...], dist.ProcessGroup]:
    """每个 broadcast 子组(``{src}∪dsts`` 的有序 rank 元组)建一个 cross-world PG。

    全 rank 按同一排序遍历(子组集由 plan 决定,确定性一致),**只有成员 rank 调入**——
    每子组独立 PrefixStore(键含成员)做 rendezvous,非成员不碰、无需参与。
    """
    timeout = datetime.timedelta(seconds=timeout_s)
    groups: dict[tuple[int, ...], dist.ProcessGroup] = {}
    for key in sorted(set(subgroup_ranks)):
        if rank not in key:
            continue
        tag = "_".join(map(str, key))
        # _new_process_group_helper 用本进程的默认-world rank 判成员(global_rank in
        # global_ranks_in_group)。子组按 cross-world 编号(vLLM=32+),与默认 world
        # (vLLM=0-31)不一致 → 会被判非成员返回 -100。把本进程默认 rank 填进自己的
        # 槽位即可过校验;rendezvous 只靠 group_rank+store,pg_group_ranks 下面自己覆盖。
        grig = list(key)
        grig[key.index(rank)] = dist.get_rank()
        pg, _ = _new_process_group_helper(
            group_size=len(key),
            group_rank=key.index(rank),
            global_ranks_in_group=grig,
            backend=backend,
            store=dist.PrefixStore(f"etha_bcast_{tag}", store),
            group_name=f"etha_bcast_{tag}",
            timeout=timeout,
        )
        _world.pg_group_ranks[pg] = {r: i for i, r in enumerate(key)}
        groups[key] = pg
    return groups
