"""Dual-endpoint (colocated) agent flow: one agent hosts both roles.

The colocated deployment puts a train-role client and an infer-role client on
the same agent. This file drives the agent directly (step loop in a helper
thread) and checks the rendezvous/batch contracts that make that work:

- InitPair never blocks the agent: the first role's command parks, the pair
  completes from the main-loop poll when the second role registers, and BOTH
  commands' semaphores release at completion;
- the two roles' RegisterTensors merge into ONE batch (never overwrite) and
  chunks generate only once both are present — the first role's semaphore
  releases then, not earlier;
- role=None on a dual pair is rejected and reported to the client;
- Transfer is per-direction and per-round: a direction-round executes only
  after BOTH roles' commands arrived (source-ready and dest-ready), a
  re-issued or older round is acknowledged without re-execution, and the
  reverse direction never auto-executes;
- an FP32 master transfers to a BF16 target bit-stably, master untouched.

The split (one role per agent) path is kept as a CPU regression: disjoint
sides, role=None, rank0 dtype exchange across agents still intact.
"""

import os
import time
import socket
import logging
import threading
from types import SimpleNamespace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import pytest
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard, Replicate

from etha.comm.ir import Transport
from etha.pg_utils import _PROCESS_GROUP_CACHE
from etha.tensor_bus import BatchHandler, TensorBusAgent, TensorBusClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAIR = "colocated_weights"
TRAIN, INFER = "train", "infer"
BATCH = "step_0"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


class _MeshSpec:
    """Client-side mesh stand-in: init_pair only serializes ``mesh.shape``.

    A real DeviceMesh would initialize a process group in the client process;
    here the only PG lives inside the agent.
    """

    def __init__(self, shape: tuple[int, ...]):
        self.mesh = SimpleNamespace(shape=tuple(shape))


def _cleanup_stale(command_queue_path: str, state_path: str) -> None:
    for path_str in (command_queue_path, state_path):
        path = Path(path_str)
        for f in path.parent.glob(f"{path.name}*"):
            f.unlink(missing_ok=True)


def _agent_loop(agent: TensorBusAgent, stop: threading.Event) -> None:
    while not stop.is_set():
        agent.step()


def test_closed_batch_handler_does_not_clean_up_a_reused_batch_id():
    calls = []

    class Client:
        def _execute_command_with_semaphore(self, command, *_args, **_kwargs):
            calls.append(command.batch_id)

    client = Client()
    handler = BatchHandler(client, batch_id="reused", pair_names=["pair"])
    handler.close()
    handler.close()
    handler.__del__()
    assert calls == ["reused"]


@pytest.mark.timeout(600)
def test_dual_endpoint_agent_flow(tmp_path, monkeypatch):
    """One agent, two in-process clients: rendezvous, merge, dedup transfer."""
    import etha.tensor_bus.agent as agent_module

    real_bucket_comm = agent_module.bucket_comm

    def checked_bucket_comm(*, buckets, sequential=False):
        assert sequential, "dual batches must preserve bucket order without an environment variable"
        return real_bucket_comm(buckets=buckets, sequential=sequential)

    monkeypatch.delenv("ARC2_TB_SEQUENTIAL", raising=False)
    monkeypatch.setattr(agent_module, "bucket_comm", checked_bucket_comm)

    env = {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(_free_port()),
    }
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)

    cmd_path = str(tmp_path / "0_command.lmdb")
    state_path = str(tmp_path / "0_state.lmdb")
    _cleanup_stale(cmd_path, state_path)

    agent = TensorBusAgent(
        rank=0,
        world_size=1,
        store_host="localhost",
        store_port=_free_port(),
        lmdb_command_queue_path=cmd_path,
        lmdb_state_path=state_path,
        dist_backend="cpu:gloo",
    )
    stop = threading.Event()
    thread: threading.Thread | None = None
    try:
        client_train = TensorBusClient(agent_rank=0, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path)
        client_infer = TensorBusClient(agent_rank=0, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path)

        # The agent steps in a helper thread throughout (clients below block).
        thread = threading.Thread(target=_agent_loop, args=(agent, stop), daemon=True)
        thread.start()

        # --- InitPair rendezvous: first role parks, does not block the agent.
        # Its blocking init_pair can only return once the SECOND role
        # registers — if the agent waited synchronously this would deadlock.
        outcome: dict[str, object] = {}

        def _init_train_blocking():
            try:
                client_train.init_pair(
                    pair_name=PAIR,
                    local_name=TRAIN,
                    remote_name=INFER,
                    expected_world_size=1,
                    device_mesh=_MeshSpec((1,)),
                    placements=(Shard(0),),
                    timeout=60,
                )
                outcome["ok"] = True
            except Exception as e:  # pragma: no cover - surfaced by the assert below
                outcome["err"] = repr(e)

        init_train_thread = threading.Thread(target=_init_train_blocking, daemon=True)
        init_train_thread.start()
        time.sleep(0.5)
        assert "ok" not in outcome, "first role's init_pair must not complete alone"
        assert PAIR in agent.pending_pairs, "first role's InitPair must park, not complete"
        assert PAIR not in agent.pairs

        client_infer.init_pair(
            pair_name=PAIR,
            local_name=INFER,
            remote_name=TRAIN,
            expected_world_size=1,
            device_mesh=_MeshSpec((1, 1)),
            placements=(Replicate(), Shard(1)),
            timeout=60,
        )
        init_train_thread.join(timeout=30)
        assert outcome.get("ok") is True, f"parked init_pair never released: {outcome}"
        assert PAIR in agent.pairs and PAIR not in agent.pending_pairs
        pair = agent.pairs[PAIR]
        assert pair.dual_endpoint, "both roles on the same rank set must mark the pair dual"
        assert set(pair.role_ranks) == {TRAIN, INFER}

        # --- dual registration: two roles merge into ONE batch
        torch.manual_seed(0)
        master = torch.randn(8, 4, dtype=torch.float32)
        master_snapshot = master.clone()
        infer_local = torch.zeros(8, 4, dtype=torch.bfloat16)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_train = pool.submit(client_train.register_tensors, batch_id=BATCH, tensors=[(master, PAIR)], role=TRAIN)
            fut_infer = pool.submit(
                client_infer.register_tensors, batch_id=BATCH, tensors=[(infer_local, PAIR)], role=INFER
            )
            handler_train = fut_train.result(timeout=60)
            handler_infer = fut_infer.result(timeout=60)

        batch = agent.batches[BATCH]
        assert batch.dual
        assert set(batch.pair_role_tensors) == {(PAIR, TRAIN), (PAIR, INFER)}, "roles merge, never overwrite"
        assert set(batch.dual_direction_buckets) == {(TRAIN, INFER), (INFER, TRAIN)}
        fwd_buckets = batch.dual_direction_buckets[(TRAIN, INFER)]
        assert any(c.transport is Transport.LOCAL for b in fwd_buckets for c in b.chunks), (
            "same-rank overlap must surface as LOCAL chunks"
        )
        assert not batch.send_buckets and not batch.recv_buckets, "dual batches never fill split fields"
        assert not batch.pending_register_semaphores, "parked registration semaphores release at completion"

        # --- role contract guard: role=None on a dual pair is rejected
        with pytest.raises(RuntimeError, match="role is required"):
            client_train.register_tensors(batch_id="bad_batch", tensors=[(master, PAIR)], role=None)
        assert set(agent.batches) == {BATCH}, "rejected registration leaves no residue"

        # --- per-direction transfer: BOTH roles' commands are required
        # (send parks until the peer role's recv arrives — issued concurrently
        # because a send blocks until the round executes)
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_send = pool.submit(handler_train.transfer, transfer_type="send", role=TRAIN, blocking=True, timeout=60)
            fut_recv = pool.submit(handler_infer.transfer, transfer_type="recv", role=INFER, blocking=True, timeout=60)
            fut_send.result(timeout=90)
            fut_recv.result(timeout=90)

        assert torch.equal(infer_local, master.to(torch.bfloat16)), "BF16 target must match the cast master"
        assert torch.equal(master, master_snapshot), "FP32 master must be bit-stable through the transfer"
        assert batch.dual_direction_done.get((TRAIN, INFER)) == 0, "round 0 executed exactly once"
        assert (INFER, TRAIN) not in batch.dual_direction_done, "reverse direction must never auto-execute"
        assert handler_train.query_transfer_signal(sync_round=0) is True, (
            "versioned round signal must be queryable once the round executed"
        )
        assert handler_train.query_transfer_signal(sync_round=1) is False, "future rounds are not signaled"

        # --- cleanup is idempotent at the handler level
        handler_train.close()
        handler_infer.close()
        assert BATCH not in agent.batches

        client_train.close()
        client_infer.close()
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=30)
        agent.close(destroy=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        # The agent's subgroup cache outlives the destroyed world; a later
        # main-process world-1 test would reuse dead handles (size-1 barriers
        # pass silently, broadcasts raise) — clear it at the world boundary.
        _PROCESS_GROUP_CACHE.clear()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


SPLIT_PAIR = "split_weights"


def _split_worker(rank: int, world_size: int, root: str, store_port: int, dist_port: int) -> None:
    """Split (one role per agent) regression on CPU: the legacy path unchanged."""
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), MASTER_ADDR="localhost", MASTER_PORT=str(dist_port))
    cmd_path = str(Path(root) / f"{rank}_command.lmdb")
    state_path = str(Path(root) / f"{rank}_state.lmdb")
    _cleanup_stale(cmd_path, state_path)

    agent = TensorBusAgent(
        rank=rank,
        world_size=world_size,
        store_host="localhost",
        store_port=store_port,
        lmdb_command_queue_path=cmd_path,
        lmdb_state_path=state_path,
        dist_backend="cpu:gloo",
    )
    stop = threading.Event()
    thread = threading.Thread(target=_agent_loop, args=(agent, stop), daemon=True)
    thread.start()
    client = TensorBusClient(agent_rank=rank, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path)
    try:
        torch.manual_seed(0)
        master = torch.randn(8, 4, dtype=torch.float32)  # seeded: identical on both ranks

        if rank == 0:  # train side: FP32 master
            client.init_pair(
                pair_name=SPLIT_PAIR,
                local_name=TRAIN,
                remote_name=INFER,
                expected_world_size=1,
                device_mesh=_MeshSpec((1,)),
                placements=(Shard(0),),
                timeout=60,
            )
            handler = client.register_tensors(batch_id=BATCH, tensors=[(master, SPLIT_PAIR)], role=None)
            handler.transfer(transfer_type="send", blocking=True, timeout=60)
        else:  # infer side: BF16 target, filled by the transfer
            client.init_pair(
                pair_name=SPLIT_PAIR,
                local_name=INFER,
                remote_name=TRAIN,
                expected_world_size=1,
                device_mesh=_MeshSpec((1,)),
                placements=(Shard(0),),
                timeout=60,
            )
            target = torch.zeros(8, 4, dtype=torch.bfloat16)
            handler = client.register_tensors(batch_id=BATCH, tensors=[(target, SPLIT_PAIR)], role=None)
            while not handler.query_transfer_signal():
                time.sleep(0.05)
            handler.transfer(transfer_type="recv", blocking=True, timeout=60)
            assert torch.equal(target, master.to(torch.bfloat16)), "split transfer mismatch"
            assert not agent.batches[BATCH].dual, "split batches must stay on the legacy path"

        handler.close()
        client.close()
    finally:
        stop.set()
        thread.join(timeout=30)
        agent.close(destroy=True)


@pytest.mark.timeout(600)
def test_split_pair_regression_cpu(tmp_path):
    """Disjoint single-role agents keep the legacy register/transfer path."""
    store_port = _free_port()
    dist_port = _free_port()
    try:
        torch.multiprocessing.spawn(
            _split_worker,
            args=(2, str(tmp_path), store_port, dist_port),
            nprocs=2,
            join=True,
        )
    except Exception as e:
        pytest.fail(f"Split-pair regression failed: {e}")
