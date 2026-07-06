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
from torch.distributed.distributed_c10d import GroupMember, _new_process_group_helper, _update_default_pg, _world

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
    parent_group: dist.ProcessGroup | None = None,
    backend: str = "nccl",
    timeout_s: float = _TIMEOUT_S,
) -> dict[tuple[int, ...], dist.ProcessGroup]:
    """每个 broadcast 子组(``{src}∪dsts`` 的有序 rank 元组)建一个 cross-world PG。

    推荐路径:所有 cross-world rank 都传入 ``parent_group``(即 ``etha_cross``),
    本函数临时把它设为 torch 默认 PG,再用稳定 group_name 调
    ``_new_process_group_helper`` 创建子组。这样成员校验、NCCL 拓扑都用
    cross-world rank,同时避免 ``dist.new_group`` 的本地计数 group_name 在
    trainer/vLLM 两侧不一致。

    兼容路径:未传 ``parent_group`` 时保留老的成员-only 手工拼接逻辑。
    """
    timeout = datetime.timedelta(seconds=timeout_s)
    groups: dict[tuple[int, ...], dist.ProcessGroup] = {}
    keys = sorted(set(subgroup_ranks))

    if parent_group is not None:
        old_default_pg = _world.default_pg
        _update_default_pg(parent_group)
        try:
            for key in keys:
                tag = "_".join(map(str, key))
                pg, _ = _new_process_group_helper(
                    group_size=len(key),
                    group_rank=key.index(rank) if rank in key else None,
                    global_ranks_in_group=list(key),
                    backend=backend,
                    store=dist.PrefixStore(f"etha_bcast_{tag}", store),
                    group_name=f"etha_bcast_{tag}",
                    timeout=timeout,
                )
                if pg != GroupMember.NON_GROUP_MEMBER:
                    _world.pg_group_ranks[pg] = {r: i for i, r in enumerate(key)}
                    groups[key] = pg
        finally:
            _update_default_pg(old_default_pg)
        return groups

    for key in keys:
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
