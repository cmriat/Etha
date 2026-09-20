"""Regression coverage for canonical broadcast-group creation."""

import os
import time
import socket
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import pytest
import torch.distributed as dist
from torch.distributed._tensor import Shard, Replicate, DeviceMesh, distribute_tensor

import etha.pg_utils as pg_utils
import etha.comm.transfer as transfer_module
import etha.tensor_bus.agent as agent_module
import etha.tensor_bus.client as client_module
from etha.comm import M2MMap, bucket_comm, get_m2m_map, m2m_to_chunks, chunk_to_bucket_ops, prewarm_broadcast_groups
from etha.comm.ir import Route, Endpoint
from etha.comm.transfer import Transport
from etha.tensor_bus.commands import command_error_key


def _route(src: int, *dsts: int, kind: Transport = Transport.BROADCAST) -> Route:
    return Route(
        src=Endpoint(rank=src, cell=(0,)),
        dsts=tuple(Endpoint(rank=dst, cell=(i,)) for i, dst in enumerate(dsts)),
        kind=kind,
    )


def test_prewarm_groups_are_globally_sorted_and_cached(monkeypatch):
    created = []
    handles = {}

    def new_group(ranks):
        key = tuple(ranks)
        created.append(key)
        handles[key] = object()
        return handles[key]

    monkeypatch.setattr(pg_utils, "_PROCESS_GROUP_CACHE", {})
    monkeypatch.setattr(pg_utils.dist, "new_group", new_group)

    prewarm_broadcast_groups(
        [
            M2MMap(routes=[_route(5, 3, 1), _route(1, 6, kind=Transport.P2P)]),
            None,
            M2MMap(routes=None),
            M2MMap(routes=[_route(7, 2, 0), _route(0, 7, 2), _route(2, 2, 5)]),
        ]
    )

    expected = [(0, 2, 7), (1, 3, 5), (2, 5)]
    assert created == expected

    used_groups = []

    def broadcast(_buffer, *, src, group, async_op):
        used_groups.append(group)

    monkeypatch.setattr(transfer_module.dist, "broadcast", broadcast)
    transfer_module._execute_broadcast(torch.zeros(1), src_rank=2, dst_ranks=(2, 5))

    assert created == expected
    assert used_groups == [handles[(2, 5)]]


def test_agent_validates_layout_and_prewarms_before_chunks(monkeypatch):
    maps = {name: (object(), object()) for name in ("pair_a", "pair_b")}
    group = object()
    agent = SimpleNamespace(
        rank=0,
        world_size=2,
        batches={},
        pairs={
            name: SimpleNamespace(
                local_ranks=[0],
                remote_ranks=[1],
                local_group=group,
                pair_group=group,
                local_is_first=True,
                m2m_send=pair_maps[0],
                m2m_recv=pair_maps[1],
                source_partial_groups=None,
            )
            for name, pair_maps in maps.items()
        },
    )
    events = []

    def gather_same(layouts, layout, **_kwargs):
        layouts[:] = [layout] * agent.world_size

    def receive_dtype(values, *_args, **_kwargs):
        values[0] = torch.float32

    monkeypatch.setattr(agent_module.dist, "all_gather_object", gather_same)
    monkeypatch.setattr(
        agent_module, "prewarm_broadcast_groups", lambda values: events.append(("prewarm", tuple(values)))
    )
    monkeypatch.setattr(agent_module, "m2m_to_chunks", lambda m2m, **_kwargs: events.append(("chunks", m2m)) or [])
    monkeypatch.setattr(agent_module.ForkingPickler, "loads", lambda _payload: torch.zeros(1))
    monkeypatch.setattr(agent_module.dist, "send_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent_module.dist, "recv_object_list", receive_dtype)
    monkeypatch.setattr(agent_module.dist, "broadcast_object_list", lambda *_args, **_kwargs: None)

    payload = memoryview(b"tensor")
    message = agent_module.RegisterTensors(
        batch_id="batch",
        tensors=[("pair_b", payload), ("pair_a", payload)],
    )
    agent_module.TensorBusAgent._handle_register_tensors(agent, message)

    expected_maps = (*maps["pair_a"], *maps["pair_b"])
    assert events == [("prewarm", expected_maps), *(("chunks", m2m) for m2m in expected_maps)]

    events.clear()
    pair_b = agent.pairs["pair_b"]
    pair_b.local_ranks, pair_b.remote_ranks = [1], [0]
    with pytest.raises(ValueError, match="Inconsistent or invalid RegisterTensors layout"):
        agent_module.TensorBusAgent._handle_register_tensors(agent, message)
    pair_b.local_ranks, pair_b.remote_ranks = [0], [1]
    assert events == []

    def gather_mismatch(layouts, layout, **_kwargs):
        layouts[:] = [layout, ("different",)]

    monkeypatch.setattr(agent_module.dist, "all_gather_object", gather_mismatch)
    with pytest.raises(ValueError, match="Inconsistent or invalid RegisterTensors layout"):
        agent_module.TensorBusAgent._handle_register_tensors(agent, message)
    assert events == []

    reported = []
    message.semaphore_name = "/registration-test"
    agent._handle_register_tensors = MethodType(agent_module.TensorBusAgent._handle_register_tensors, agent)
    agent._record_command_error = lambda name, error: reported.append(("error", name, str(error)))
    agent._release_semaphore = lambda name: reported.append(("release", name))
    agent_module.TensorBusAgent._execute_command(agent, message)
    assert reported[0][:2] == ("error", message.semaphore_name)
    assert reported[1] == ("release", message.semaphore_name)


def test_empty_registration_reaches_agent_error_channel(monkeypatch):
    state = {}

    class Transaction:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, key):
            return state.get(key)

        def put(self, key, value):
            state[key] = value

    class StateEnv:
        def begin(self, **_kwargs):
            return Transaction()

    class Semaphore:
        def __init__(self, *_args, **_kwargs):
            pass

        def acquire(self, **_kwargs):
            pass

        def close(self):
            pass

    writer = SimpleNamespace(state_env=StateEnv(), state_db=object(), _command_error_times={})

    class Queue:
        message = None

        def enqueue(self, message):
            self.message = message
            agent_module.TensorBusAgent._record_command_error(
                writer,
                message.semaphore_name,
                agent_module._InvalidRegistrationError("empty registration"),
            )

    queue = Queue()
    client = client_module.TensorBusClient.__new__(client_module.TensorBusClient)
    client.agent_rank = 0
    client.command_queue = queue
    client.state_env = writer.state_env
    client.state_db = writer.state_db
    monkeypatch.setattr(client_module.posix_ipc, "Semaphore", Semaphore)

    with pytest.raises(RuntimeError, match="register_tensors failed.*empty registration"):
        client.register_tensors("batch", [], timeout=1)
    assert queue.message.tensors == []


def test_command_error_record_lifecycle(monkeypatch):
    state = {}

    class Transaction:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def put(self, key, value):
            state[key] = value

        def delete(self, key):
            state.pop(key, None)

    class StateEnv:
        def begin(self, **_kwargs):
            return Transaction()

    agent = SimpleNamespace(
        rank=0,
        state_env=StateEnv(),
        state_db=object(),
        _command_error_times={},
        pairs={},
        batches={},
    )
    record = agent_module.TensorBusAgent._record_command_error

    monkeypatch.setattr(agent_module, "COMMAND_ERROR_TTL", 0.0)
    record(agent, "/sem-a", ValueError("first"))
    record(agent, "/sem-b", ValueError("second"))
    assert command_error_key("/sem-a") not in state
    assert command_error_key("/sem-b") in state

    released = []

    def fail_record(*_args):
        raise RuntimeError("state lmdb unavailable")

    def fail_registration(_msg):
        raise agent_module._InvalidRegistrationError("bad layout")

    agent._record_command_error = fail_record
    agent._release_semaphore = released.append
    agent._handle_register_tensors = fail_registration
    message = agent_module.RegisterTensors(batch_id="batch", tensors=[])
    message.semaphore_name = "/sem-c"
    agent_module.TensorBusAgent._execute_command(agent, message)
    assert released == ["/sem-c"]


def _run_bidirectional_broadcast(rank: int):
    world_size = 8
    mesh_size = world_size // 2
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    mesh_a = DeviceMesh("cpu", torch.arange(mesh_size).view(2, 2))
    mesh_b = DeviceMesh("cpu", torch.arange(mesh_size, world_size).view(2, 2))
    specs_a = [Shard(0), Replicate()]
    specs_b = [Replicate(), Shard(0)]
    tensor_a = torch.arange(64, dtype=torch.float32).view(8, 8)
    tensor_b = -tensor_a - 1
    is_a = rank < mesh_size

    if is_a:
        source = distribute_tensor(tensor_a, mesh_a, specs_a).to_local()
        reverse_target = distribute_tensor(torch.zeros_like(tensor_b), mesh_a, specs_a)
    else:
        source = distribute_tensor(tensor_b, mesh_b, specs_b).to_local()
        forward_target = distribute_tensor(torch.zeros_like(tensor_a), mesh_b, specs_b)

    a_to_b = get_m2m_map(mesh_a, specs_a, mesh_b, specs_b, group=dist.group.WORLD, device="cpu")
    b_to_a = get_m2m_map(mesh_b, specs_b, mesh_a, specs_a, group=dist.group.WORLD, device="cpu")
    assert any(route.kind == Transport.BROADCAST for route in a_to_b.routes or ())
    assert any(route.kind == Transport.BROADCAST for route in b_to_a.routes or ())

    prewarm_broadcast_groups([a_to_b, b_to_a])
    if is_a:
        send = m2m_to_chunks(a_to_b, rank=rank, source_tensor=source)
        recv = m2m_to_chunks(b_to_a, rank=rank, target_tensor=reverse_target.to_local())
    else:
        send = m2m_to_chunks(b_to_a, rank=rank, source_tensor=source)
        recv = m2m_to_chunks(a_to_b, rank=rank, target_tensor=forward_target.to_local())

    with patch("etha.comm.comm_methods.torch.cuda.synchronize"):
        bucket_comm(chunk_to_bucket_ops(send if is_a else recv, bucket_size=1))
        dist.barrier()
        bucket_comm(chunk_to_bucket_ops(recv if is_a else send, bucket_size=1))

    if is_a:
        assert torch.equal(reverse_target.full_tensor(), tensor_b)
    else:
        assert torch.equal(forward_target.full_tensor(), tensor_a)

    ranks_a = list(range(mesh_size))
    ranks_b = list(range(mesh_size, world_size))
    local_ranks, remote_ranks = (ranks_a, ranks_b) if is_a else (ranks_b, ranks_a)
    invalid_agent = SimpleNamespace(
        rank=rank,
        world_size=world_size,
        batches={},
        pairs={"pair": SimpleNamespace(local_ranks=local_ranks, remote_ranks=remote_ranks)},
    )
    invalid_tensors = [] if rank == 0 else [("pair", memoryview(b"tensor"))]
    invalid_message = agent_module.RegisterTensors(batch_id="invalid", tensors=invalid_tensors)
    with pytest.raises(ValueError, match="Inconsistent or invalid RegisterTensors layout"):
        agent_module.TensorBusAgent._handle_register_tensors(invalid_agent, invalid_message)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_bidirectional_broadcast_group_order():
    os.environ["MASTER_ADDR"] = "localhost"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        os.environ["MASTER_PORT"] = str(sock.getsockname()[1])

    context = torch.multiprocessing.spawn(_run_bidirectional_broadcast, nprocs=8, join=False)
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("distributed workers did not finish within 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        cleanup_deadline = time.monotonic() + 5
        for process in context.processes:
            process.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join()
