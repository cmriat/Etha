"""Regression tests for canonical broadcast-group creation order.

``dist.new_group`` is collective on WORLD: every rank must issue the same
sequence of calls, non-members included. ``m2m_to_chunks`` creates a
direction's broadcast groups on first touch, and the two sides of a pair
materialize chunks in opposite direction orders (each walks local-send before
local-recv). When BOTH directions broadcast — e.g. train side with
``dp_replicate >= 2`` against replicated TP1 inference — the per-direction
creations interleave differently on the two sides and the communicators
silently cross-wire (wrong data) or hang. ``prewarm_broadcast_groups`` creates
the union in one globally sorted pass so both sides agree.
"""

import os
import time
import socket
import logging
from contextlib import nullcontext
from unittest.mock import patch
from multiprocessing.reduction import ForkingPickler

import torch
import pytest
import torch.distributed as dist
from torch.distributed._tensor import Shard, Replicate, DeviceMesh, distribute_tensor

import etha.pg_utils as pg_utils
import etha.comm.transfer as transfer_module
import etha.comm.get_chunks as get_chunks_module
import etha.tensor_bus.agent as agent_module
from etha.comm import (
    M2MMap,
    bucket_comm,
    get_m2m_map,
    m2m_to_chunks,
    chunk_to_bucket_ops,
    broadcast_group_ranks,
    prewarm_broadcast_groups,
)
from etha.comm.ir import Route, Endpoint
from etha.comm.transfer import Transport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_broadcast_group_ranks_canonicalizes():
    """Broadcast groups use sorted full membership; P2P contributes none."""
    bcast = Route(
        src=Endpoint(rank=5, cell=(0,)),
        dsts=(Endpoint(rank=4, cell=(0, 0)), Endpoint(rank=0, cell=(0, 1))),
        kind=Transport.BROADCAST,
    )
    bcast_same_members_other_root = Route(
        src=Endpoint(rank=0, cell=(1,)),
        dsts=(Endpoint(rank=5, cell=(1, 0)), Endpoint(rank=4, cell=(1, 1))),
        kind=Transport.BROADCAST,
    )
    p2p = Route(
        src=Endpoint(rank=1, cell=(0,)),
        dsts=(Endpoint(rank=6, cell=(0, 0)),),
        kind=Transport.P2P,
    )
    m2m = M2MMap(routes=[bcast, bcast_same_members_other_root, p2p])
    assert broadcast_group_ranks(m2m) == {(0, 4, 5)}
    assert broadcast_group_ranks(M2MMap(routes=None)) == set()


def test_prewarm_broadcast_groups_sorts_union(monkeypatch):
    """Map and route order cannot affect the process-group creation sequence."""
    group_a = Route(
        src=Endpoint(rank=7, cell=(0,)),
        dsts=(Endpoint(rank=2, cell=(0, 0)), Endpoint(rank=0, cell=(0, 1))),
        kind=Transport.BROADCAST,
    )
    group_a_other_root = Route(
        src=Endpoint(rank=0, cell=(1,)),
        dsts=(Endpoint(rank=7, cell=(1, 0)), Endpoint(rank=2, cell=(1, 1))),
        kind=Transport.BROADCAST,
    )
    group_b = Route(
        src=Endpoint(rank=5, cell=(0,)),
        dsts=(Endpoint(rank=3, cell=(0, 0)), Endpoint(rank=1, cell=(0, 1))),
        kind=Transport.BROADCAST,
    )
    calls = []
    monkeypatch.setattr(get_chunks_module, "get_or_create_process_group", lambda ranks: calls.append(tuple(ranks)))

    prewarm_broadcast_groups(
        [
            M2MMap(routes=[group_b]),
            None,
            M2MMap(routes=[group_a_other_root, group_a]),
        ]
    )

    assert calls == [(0, 2, 7), (1, 3, 5)]


def test_prewarm_and_execute_share_group_cache(monkeypatch):
    """A source that is also a destination must not create a second group."""
    route = Route(
        src=Endpoint(rank=2, cell=(0,)),
        dsts=(Endpoint(rank=2, cell=(0, 0)), Endpoint(rank=5, cell=(0, 1))),
        kind=Transport.BROADCAST,
    )
    created = []
    broadcasts = []
    group = object()
    work = object()

    monkeypatch.setattr(pg_utils, "_PROCESS_GROUP_CACHE", {})
    monkeypatch.setattr(
        pg_utils.dist,
        "new_group",
        lambda ranks: created.append(tuple(ranks)) or group,
    )

    def record_broadcast(_buffer, *, src, group, async_op):
        broadcasts.append((src, group, async_op))
        return work

    monkeypatch.setattr(transfer_module.dist, "broadcast", record_broadcast)

    prewarm_broadcast_groups([M2MMap(routes=[route])])
    result = transfer_module._execute_broadcast(torch.zeros(1), src_rank=2, dst_ranks=(2, 5))

    assert created == [(2, 5)]
    assert broadcasts == [(2, group, True)]
    assert result is work


def test_agent_prewarms_all_pair_directions_before_chunks(monkeypatch):
    """The Agent wires every pair's send/recv maps into prewarm first."""
    maps = {name: (M2MMap(routes=[]), M2MMap(routes=[])) for name in ("pair_a", "pair_b")}
    pair_group = object()
    local_group = object()
    agent = object.__new__(agent_module.TensorBusAgent)
    agent.rank = 0
    agent.pairs = {
        name: agent_module.PairState(
            pair_name=name,
            local_name="a",
            local_ranks=[0],
            remote_name="b",
            remote_ranks=[1],
            pair_size=2,
            local_group=local_group,
            pair_group=pair_group,
            local_is_first=True,
            m2m_send=pair_maps[0],
            m2m_recv=pair_maps[1],
        )
        for name, pair_maps in maps.items()
    }
    agent.batches = {}
    events = []

    def record_prewarm(m2m_maps):
        events.append(("prewarm", tuple(m2m_maps)))

    def record_chunks(m2m, **_kwargs):
        events.append(("chunks", m2m))
        return []

    def receive_dtype(values, *_args, **_kwargs):
        values[0] = torch.float32

    monkeypatch.setattr(agent_module, "prewarm_broadcast_groups", record_prewarm)
    monkeypatch.setattr(agent_module, "m2m_to_chunks", record_chunks)
    monkeypatch.setattr(ForkingPickler, "loads", lambda _payload: torch.zeros(1))
    monkeypatch.setattr(agent_module.dist, "send_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent_module.dist, "recv_object_list", receive_dtype)
    monkeypatch.setattr(agent_module.dist, "broadcast_object_list", lambda *_args, **_kwargs: None)

    payload = memoryview(b"tensor")
    message = agent_module.RegisterTensors(
        batch_id="batch",
        tensors=[("pair_b", payload), ("pair_a", payload)],
    )
    agent._handle_register_tensors(message)

    expected_maps = (*maps["pair_b"], *maps["pair_a"])
    assert events[0] == ("prewarm", expected_maps)
    assert tuple(event[1] for event in events[1:]) == expected_maps


def run_bidirectional_broadcast(rank: int, world_size: int, device: str):
    """Exercise the TensorBus registration order with broadcasts in both directions.

    Mirrors ``TensorBusAgent._handle_register_tensors``: both maps are generated
    in the same order on every rank, then each side materializes its
    send-direction chunks before its recv-direction chunks — opposite
    directions on the two sides. Without the prewarm, per-direction first-touch
    group creation interleaves the WORLD-collective ``new_group`` calls
    differently per side and the transfers deliver wrong data (or hang).
    """
    dist.init_process_group(backend="nccl" if device == "cuda" else "gloo", rank=rank, world_size=world_size)
    mesh_size = world_size // 2
    assert mesh_size == 4

    # [Shard(0), Replicate()] on A vs [Replicate(), Shard(0)] on B: each side
    # holds every shard replicated across 2 ranks, so BOTH directions fan out
    # from a single chosen source to 2 targets — the dp_replicate>=2 topology.
    mesh_a = DeviceMesh(device, torch.arange(mesh_size).view(2, 2))
    mesh_b = DeviceMesh(device, torch.arange(mesh_size, mesh_size * 2).view(2, 2))
    specs_a = [Shard(0), Replicate()]
    specs_b = [Replicate(), Shard(0)]

    torch.manual_seed(0)
    tensor_a_origin = torch.randn(64, 64, device=device)
    torch.manual_seed(1)
    tensor_b_origin = torch.randn(64, 64, device=device)
    is_a = rank < mesh_size

    if is_a:
        src_local = distribute_tensor(tensor_a_origin, mesh_a, specs_a).to_local()
        rev_target = distribute_tensor(torch.zeros_like(tensor_b_origin), mesh_a, specs_a)
    else:
        src_local = distribute_tensor(tensor_b_origin, mesh_b, specs_b).to_local()
        fwd_target = distribute_tensor(torch.zeros_like(tensor_a_origin), mesh_b, specs_b)

    # Both directions' maps, generated in the same order on every rank
    # (mirrors init_pair's canonical first/second discipline).
    m2m_a_to_b = get_m2m_map(
        source_mesh=mesh_a,
        source_placements=specs_a,
        target_mesh=mesh_b,
        target_placements=specs_b,
        group=dist.group.WORLD,
        device=device,
    )
    m2m_b_to_a = get_m2m_map(
        source_mesh=mesh_b,
        source_placements=specs_b,
        target_mesh=mesh_a,
        target_placements=specs_a,
        group=dist.group.WORLD,
        device=device,
    )

    # The topology under test: broadcasts in BOTH directions.
    assert broadcast_group_ranks(m2m_a_to_b), "forward direction must broadcast"
    assert broadcast_group_ranks(m2m_b_to_a), "reverse direction must broadcast"

    # The fix: create the union of broadcast groups in one canonical pass
    # before either side materializes chunks.
    prewarm_broadcast_groups([m2m_a_to_b, m2m_b_to_a])

    # Per-side registration order, as in TensorBusAgent._handle_register_tensors:
    # BOTH sides materialize their local send direction first, then their local
    # recv direction — which are opposite directions on the two sides.
    if is_a:
        send_chunks = m2m_to_chunks(m2m_a_to_b, rank=rank, source_tensor=src_local)
        recv_chunks = m2m_to_chunks(m2m_b_to_a, rank=rank, target_tensor=rev_target.to_local())
    else:
        send_chunks = m2m_to_chunks(m2m_b_to_a, rank=rank, source_tensor=src_local)
        recv_chunks = m2m_to_chunks(m2m_a_to_b, rank=rank, target_tensor=fwd_target.to_local())

    # bucket_comm's final CUDA fence is unrelated to this CPU/Gloo regression.
    sync_context = patch("etha.comm.comm_methods.torch.cuda.synchronize") if device == "cpu" else nullcontext()
    with sync_context:
        # Forward transfer: A sends, B receives.
        bucket_comm(buckets=chunk_to_bucket_ops(chunks=send_chunks if is_a else recv_chunks, bucket_size=1))
        dist.barrier()

        # Reverse transfer: B sends, A receives.
        bucket_comm(buckets=chunk_to_bucket_ops(chunks=recv_chunks if is_a else send_chunks, bucket_size=1))

    if is_a:
        assert torch.allclose(rev_target.full_tensor(), tensor_b_origin), "reverse transfer corrupted"
    else:
        assert torch.allclose(fwd_target.full_tensor(), tensor_a_origin), "forward transfer corrupted"

    dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_bidirectional_broadcast_group_order():
    """Two 4-rank meshes, broadcasts in both directions, opposite per-side order."""
    world_size = 8
    device = "cpu"

    os.environ["MASTER_ADDR"] = "localhost"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    os.environ["MASTER_PORT"] = str(port)

    process_context = torch.multiprocessing.spawn(
        run_bidirectional_broadcast,
        args=(world_size, device),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not process_context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("distributed workers did not finish within 90 seconds")
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
        cleanup_deadline = time.monotonic() + 5
        for process in process_context.processes:
            process.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        for process in process_context.processes:
            if process.is_alive():
                process.kill()
                process.join()
