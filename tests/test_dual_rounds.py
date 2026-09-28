"""Dual-endpoint sync rounds, multi-pair batches, and pair-init ordering.

Regressions for the round-handshake Criticals, all CPU:

1. ``test_multi_round_same_batch_updates_every_round`` — a dual batch
   supports unbounded sync rounds on the SAME batch: round r executes only
   when both roles' Transfer commands for r arrived, updates the target
   every round (a permanent done-flag would freeze round 1+), acknowledges
   stale (r < done) and duplicate (r == done) rounds WITHOUT re-executing
   them, and publishes a versioned per-round completion signal.

2. ``test_multi_pair_dual_batch_transfers_every_pair`` — a batch spanning
   two pairs with different placements: each direction's chunks accumulate
   across ALL pairs before bucketization (per-pair assignment kept only the
   last pair's weights). Multiple tensors per pair, every round writes
   distinct global coordinates, cross-rank reversed command order in one
   round.

3. ``test_pair_init_out_of_order_no_deadlock`` — two agents whose InitPair
   commands arrive in different pair orders: pending pairs must complete in
   one globally serialized sequence (the store-backed completion log) so
   new_group bootstraps cannot interleave; both pairs end up usable.

4. ``test_request_signal_drives_reactive_recv`` — the PRODUCTION
   notification path, not a hand-staged concurrent send/recv: the trainer
   issues ONLY its blocking send; a watcher (the engine service's stand-in)
   polls query_transfer_request and issues the round's recv only after the
   request is published. The request is visible while the send is still
   parked, the COMPLETION signal is not, and driving a recv off completion
   would deadlock — this pins the request/completion split.

5. ``test_pair_completion_order_identical_across_ranks`` — world-6, two
   dual pairs whose mesh shapes ((2,3) vs (3,2)) create disjoint REAL mesh
   sub-groups (the only surface where per-rank group-creation order can
   diverge), pair NAME order opposing readiness order, and one rank whose
   completion polling starts only after both pairs are ready. Every rank
   must complete pairs in the IDENTICAL sequence (the store-backed
   completion log) and both pairs must stay usable through one spanning
   dual batch. Pre-log, order divergence across ranks is a race (the
   all-rank rendezvous of a completed pair usually forces late ranks into
   the same order); this test pins the invariant the log guarantees.
"""

import os
import json
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

from etha.pg_utils import _PROCESS_GROUP_CACHE
from etha.tensor_bus import TensorBusAgent, TensorBusClient
from etha.tensor_bus.batch_state import BatchState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN, INFER = "train", "infer"


def _direction_store(values=None):
    values = {} if values is None else values

    def get(key, *, component):
        return values.get((component, key))

    def set_value(key, value, *, component):
        values[(component, key)] = value.encode()

    def wait_for_key(key, *, timeout, component):
        return values[(component, key)]

    return SimpleNamespace(get=get, set=set_value, wait_for_key=wait_for_key)


@pytest.mark.parametrize(("source", "target"), [(TRAIN, INFER), ("actor", "serve")])
def test_dual_batch_pins_first_direction_with_configurable_roles(source, target):
    agent = TensorBusAgent.__new__(TensorBusAgent)
    agent.rank = 0
    agent.store = _direction_store()
    agent.pairs = {"p": SimpleNamespace(role_ranks={source: [0], target: [0]}, pair_name="p")}
    agent._publish_transfer_request = lambda *_args: None
    batch = BatchState(batch_id="b", pair_names=["p"], dual=True, local_leader=0)
    first = SimpleNamespace(batch_id="b", role=source, transfer_type="send", sync_round=0, semaphore_name=None)
    assert agent._execute_dual_transfer(first, batch) is False
    assert batch.dual_direction == (source, target)

    reverse = SimpleNamespace(batch_id="b", role=target, transfer_type="send", sync_round=0)
    with pytest.raises(ValueError, match="one direction only"):
        agent._execute_dual_transfer(reverse, batch)


def test_partial_target_direction_fails_before_collectives(monkeypatch):
    agent = TensorBusAgent.__new__(TensorBusAgent)
    agent.rank = 0
    agent.store = _direction_store()
    agent.pairs = {"p": SimpleNamespace(role_ranks={TRAIN: [0], INFER: [0]}, pair_name="p")}
    agent._publish_transfer_request = lambda *_args: None
    batch = BatchState(batch_id="b", pair_names=["p"], dual=True, local_leader=0)
    batch.dual_direction_buckets[(TRAIN, INFER)] = None
    send = SimpleNamespace(batch_id="b", role=TRAIN, transfer_type="send", sync_round=0, semaphore_name=None)
    recv = SimpleNamespace(batch_id="b", role=INFER, transfer_type="recv", sync_round=0, semaphore_name=None)
    assert agent._execute_dual_transfer(send, batch) is False
    monkeypatch.setattr(dist, "barrier", lambda _group: pytest.fail("collective reached for Partial target"))
    with pytest.raises(ValueError, match="Partial target"):
        agent._execute_dual_transfer(recv, batch)


def test_init_pair_rejects_mismatched_remote_peer():
    from etha.tensor_bus.commands import InitPair

    agent = TensorBusAgent.__new__(TensorBusAgent)
    agent.rank = 0
    agent.pairs = {}
    agent.pending_pairs = {}
    agent._write_init_pair_keys = lambda _msg: None
    first = InitPair(pair_name="p", local_name="A", remote_name="B", expected_world_size=1)
    agent._handle_init_pair(first)
    mismatch = InitPair(pair_name="p", local_name="B", remote_name="C", expected_world_size=1)
    with pytest.raises(ValueError, match="known: A, B"):
        agent._handle_init_pair(mismatch)
    assert "p" in agent.pending_pairs
    assert agent.pending_pairs["p"].remote_name == "B"


def test_duplicate_init_pair_rejects_conflicting_peers():
    from etha.tensor_bus.commands import InitPair
    from etha.tensor_bus.pair_state import PairState

    agent = TensorBusAgent.__new__(TensorBusAgent)
    agent.rank = 0
    agent.pairs = {
        "p": PairState(
            pair_name="p",
            local_name="A",
            local_ranks=[0],
            remote_name="B",
            remote_ranks=[0],
            pair_size=1,
            local_group=object(),
            pair_group=object(),
            local_is_first=True,
            dual_endpoint=True,
            role_ranks={"A": [0], "B": [0]},
        )
    }
    agent._write_init_pair_keys = lambda _msg: None
    mismatch = InitPair(pair_name="p", local_name="B", remote_name="C", expected_world_size=1)
    with pytest.raises(ValueError, match="known: A, B"):
        agent._handle_init_pair(mismatch)


def test_unknown_transfer_role_is_rejected_before_direction_pin():
    agent = TensorBusAgent.__new__(TensorBusAgent)
    agent.rank = 0
    agent.store = _direction_store()
    agent.pairs = {"p": SimpleNamespace(role_ranks={TRAIN: [0], INFER: [0]}, pair_name="p")}
    agent._publish_transfer_request = lambda *_args: None
    batch = BatchState(batch_id="b", pair_names=["p"], dual=True, local_leader=0)
    bogus = SimpleNamespace(batch_id="b", role="other", transfer_type="send", sync_round=0, semaphore_name=None)
    with pytest.raises(ValueError, match="is not a peer"):
        agent._execute_dual_transfer(bogus, batch)
    assert batch.dual_direction is None


def test_stale_round_ack_releases_a_parked_peer():
    agent = TensorBusAgent.__new__(TensorBusAgent)
    agent.rank = 0
    agent.store = _direction_store()
    agent.pairs = {"p": SimpleNamespace(role_ranks={TRAIN: [0], INFER: [0]}, pair_name="p")}
    released = []
    agent._release_semaphore = released.append
    batch = BatchState(batch_id="b", pair_names=["p"], dual=True, local_leader=0, dual_direction=(TRAIN, INFER))
    batch.dual_direction_done[(TRAIN, INFER)] = 2
    batch.dual_round_pending[((TRAIN, INFER), 1)] = {"roles": {TRAIN}, "semaphores": ["/parked"]}
    late = SimpleNamespace(batch_id="b", role=INFER, transfer_type="recv", sync_round=1, semaphore_name="/late")
    assert agent._execute_dual_transfer(late, batch) is True
    assert released == ["/parked"]
    assert ((TRAIN, INFER), 1) not in batch.dual_round_pending


def test_dual_layout_change_aborts_parked_register(monkeypatch):
    from etha.tensor_bus.commands import RegisterTensors

    agent = TensorBusAgent.__new__(TensorBusAgent)
    agent.rank = 0
    agent.world_size = 1
    agent.pairs = {
        "p": SimpleNamespace(role_ranks={TRAIN: [0], INFER: [0]}, local_ranks=[0], remote_ranks=[0], dual_endpoint=True)
    }
    membership = tuple(sorted(((0,), (0,))))
    batch = BatchState(batch_id="b", pair_names=["p"], dual=True, bucket_size=1)
    batch.dual_layout_signature = ("b", 1, (("p", 1, membership),), True)
    batch.pending_register_semaphores = ["/sem-train"]
    agent.batches = {"b": batch}
    released, recorded = [], []
    agent._release_semaphore = released.append
    agent._record_command_error = lambda name, _error: recorded.append(name)

    def gather(out, obj, group=None):
        del group
        out[0] = obj

    monkeypatch.setattr(dist, "all_gather_object", gather)
    msg = RegisterTensors(batch_id="b", tensors=[("p", memoryview(b"x"))], bucket_size=2, role=INFER)
    with pytest.raises(ValueError, match="layout changed"):
        agent._handle_register_tensors(msg)
    assert released == ["/sem-train"]
    assert recorded == ["/sem-train"]
    assert "b" not in agent.batches


def test_failed_dual_generation_wakes_the_parked_register():
    agent = TensorBusAgent.__new__(TensorBusAgent)
    agent.rank = 0
    agent.pairs = {
        "p": SimpleNamespace(
            role_ranks={TRAIN: [0], INFER: [0]},
            role_groups={TRAIN: object(), INFER: object()},
            pair_group=object(),
            pair_name="p",
        )
    }
    batch = BatchState(batch_id="b", pair_names=["p"], dual=True, local_leader=0)
    batch.pair_role_tensors[("p", TRAIN)] = [torch.zeros(2)]
    batch.pair_role_dtypes[("p", TRAIN)] = [torch.float32]
    batch.pending_register_semaphores = ["/sem-train"]
    agent.batches = {"b": batch}
    released, recorded = [], []
    agent._release_semaphore = released.append
    agent._record_command_error = lambda name, _error: recorded.append(name)
    agent._generate_dual_buckets = lambda _batch: (_ for _ in ()).throw(ValueError("bad layout"))
    with pytest.raises(ValueError, match="bad layout"):
        agent._register_dual_role(batch, {"p": []}, INFER, None)
    assert released == ["/sem-train"]
    assert recorded == ["/sem-train"]
    assert "b" not in agent.batches


def test_dual_batch_uses_shared_direction_before_bucket_collectives(monkeypatch):
    agent = TensorBusAgent.__new__(TensorBusAgent)
    agent.rank = 1
    agent.pairs = {"p": SimpleNamespace(role_ranks={"actor": [0, 1], "serve": [0, 1]}, pair_name="p")}
    key = "batch:b/g0/state:dual_direction"
    agent.store = _direction_store({("global", key): json.dumps(("actor", "serve")).encode()})
    batch = BatchState(batch_id="b", pair_names=["p"], dual=True, local_leader=0)
    monkeypatch.setattr(dist, "barrier", lambda _group: pytest.fail("collective reached after direction mismatch"))
    reverse = SimpleNamespace(batch_id="b", role="serve", transfer_type="send", sync_round=0)
    with pytest.raises(ValueError, match="one direction only"):
        agent._execute_dual_transfer(reverse, batch)


def test_failed_pair_completion_wakes_all_waiters():
    agent = TensorBusAgent.__new__(TensorBusAgent)
    agent.rank = 0
    agent.store = SimpleNamespace(
        get=lambda key: b"pair" if key == "pair_completion:entry:1" else None,
        set=lambda *_args, **_kwargs: None,
    )
    agent._completion_cursor = 0

    class Pending:
        def __init__(self):
            self.semaphores = {"a": ["first", "second"]}

        def all_semaphores(self):
            return [s for names in self.semaphores.values() for s in names if s]

    pending = Pending()
    agent.pending_pairs = {"pair": pending}
    recorded, released = [], []

    def fail_pair(_pending):
        raise ValueError("bad mesh")

    def record(name, error):
        recorded.append((name, str(error)))
        if name == "first":
            raise OSError("LMDB unavailable")

    agent._complete_pair = fail_pair
    agent._record_command_error = record
    agent._release_semaphore = released.append
    agent._consume_completion_log()
    assert recorded == [("first", "bad mesh"), ("second", "bad mesh")]
    assert released == ["first", "second"]
    assert agent.pending_pairs["pair"] is pending
    assert pending.all_semaphores() == []


def test_stale_cleanup_does_not_delete_a_reused_batch():
    from etha.tensor_bus.commands import CleanupBatch

    agent = TensorBusAgent.__new__(TensorBusAgent)
    agent.rank = 0
    live = BatchState(batch_id="b", pair_names=["p"], dual=True, generation=2)
    agent.batches = {"b": live}
    agent._handle_cleanup_batch(CleanupBatch(batch_id="b", generation=1))
    assert agent.batches["b"] is live


def test_dual_pair_missing_mesh_fails_before_group_creation(monkeypatch):
    import etha.tensor_bus.agent as agent_module

    agent = TensorBusAgent.__new__(TensorBusAgent)
    agent.rank = 0
    agent.pairs = {}
    box = {}
    agent.store = SimpleNamespace(
        get=lambda key, **_k: box.get(key),
        set=lambda key, value, **_k: box.__setitem__(key, value.encode() if isinstance(value, str) else value),
        wait_for_key=lambda key, **_k: box[key],
    )
    agent._check_side_ready = lambda _pair, _name: (1, [0])
    agent._collect_mesh_placement_info = lambda _pair, _ranks, _name: []
    monkeypatch.setattr(
        agent_module, "get_or_create_process_group", lambda _ranks: pytest.fail("created group for invalid pair")
    )
    pending = SimpleNamespace(pair_name="pair", local_name="actor", remote_name="serve")

    with pytest.raises(ValueError, match="requires mesh/placement"):
        agent._complete_pair(pending)
    assert agent.pairs == {}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


class _MeshSpec:
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


def _round_transfer(handler_src, handler_dst, src_role, dst_role, rnd, send_first=True):
    """Both roles' commands for one round; order controlled by send_first."""
    order = [("send", handler_src, src_role), ("recv", handler_dst, dst_role)]
    if not send_first:
        order.reverse()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(h.transfer, transfer_type=t, role=r, blocking=True, timeout=60, sync_round=rnd)
            for t, h, r in order
        ]
        for fut in futures:
            fut.result(timeout=90)


@pytest.mark.timeout(300)
def test_multi_round_same_batch_updates_every_round(tmp_path):
    """Rounds on ONE batch: park-handshake, every round lands, stale/dup skip, cleanup fails closed."""
    root = str(tmp_path)
    store_port, dist_port = _free_port(), _free_port()
    if dist.is_initialized():  # main-process tests must not stack PGs
        dist.destroy_process_group()
    # the agent constructor initializes the default PG from these env vars
    os.environ.update(RANK="0", WORLD_SIZE="1", MASTER_ADDR="localhost", MASTER_PORT=str(dist_port))
    cmd_path, state_path = f"{root}/command.lmdb", f"{root}/state.lmdb"
    _cleanup_stale(cmd_path, state_path)

    agent = TensorBusAgent(
        rank=0,
        world_size=1,
        store_host="localhost",
        store_port=store_port,
        lmdb_command_queue_path=cmd_path,
        lmdb_state_path=state_path,
        dist_backend="cpu:gloo",
    )
    stop = threading.Event()
    thread = threading.Thread(target=_agent_loop, args=(agent, stop), daemon=True)
    thread.start()
    try:
        pair = "w"
        client_train = TensorBusClient(agent_rank=0, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path)
        client_infer = TensorBusClient(agent_rank=0, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path)
        # both roles' InitPair commands concurrently: the first parks until the
        # pair rendezvous completes, which needs the second role's keys — a
        # sequential same-thread pair would self-deadlock (same contract as
        # registration below)
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_train = pool.submit(
                client_train.init_pair, pair, TRAIN, INFER, 1, _MeshSpec((1,)), (Shard(0),), timeout=60
            )
            fut_infer = pool.submit(
                client_infer.init_pair, pair, INFER, TRAIN, 1, _MeshSpec((1,)), (Replicate(),), timeout=60
            )
            fut_train.result(timeout=90)
            fut_infer.result(timeout=90)

        torch.manual_seed(0)
        master = torch.randn(8, 4, dtype=torch.float32)
        target = torch.zeros(8, 4, dtype=torch.float32)
        # both roles' registrations concurrently: the first parks until the
        # second arrives, so a sequential same-thread pair would self-deadlock
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_train = pool.submit(
                client_train.register_tensors, batch_id="b", tensors=[(master, pair)], role=TRAIN, timeout=60
            )
            fut_infer = pool.submit(
                client_infer.register_tensors, batch_id="b", tensors=[(target, pair)], role=INFER, timeout=60
            )
            handler_train = fut_train.result(timeout=90)
            handler_infer = fut_infer.result(timeout=90)

        # Round r: distinct global coordinates; rounds alternate which role's
        # command arrives first (sender-first AND recv-first must both work).
        for rnd in range(3):
            master.copy_(torch.full_like(master, float(rnd + 1)) + torch.arange(8).unsqueeze(1) * 10.0)
            expected = master.clone()
            _round_transfer(handler_train, handler_infer, TRAIN, INFER, rnd, send_first=(rnd % 2 == 0))
            assert torch.equal(target, expected), f"round {rnd}: target must track the round's values"
            assert agent.batches["b"].dual_direction_done[(TRAIN, INFER)] == rnd
            assert handler_infer.query_transfer_signal(sync_round=rnd) is True
            assert handler_infer.query_transfer_signal(sync_round=rnd + 1) is False

        # Stale round (0 < done=2) and duplicate of the executed round: both
        # acknowledged without re-execution — a poison source must NOT land.
        last = 2
        expected = target.clone()
        master.copy_(torch.full_like(master, -777.0))
        handler_train.transfer(transfer_type="send", role=TRAIN, blocking=True, timeout=60, sync_round=0)
        handler_infer.transfer(transfer_type="recv", role=INFER, blocking=True, timeout=60, sync_round=0)
        handler_train.transfer(transfer_type="send", role=TRAIN, blocking=True, timeout=60, sync_round=last)
        assert torch.equal(target, expected), "stale/duplicate rounds must not overwrite the target"
        assert not agent.batches["b"].dual_round_pending, "acknowledged rounds leave nothing parked"

        # A round still parked when the batch is cleaned up fails CLOSED: the
        # parked blocking call raises instead of returning as if it executed.
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(
                handler_train.transfer, transfer_type="send", role=TRAIN, blocking=True, timeout=60, sync_round=3
            )
            deadline = time.monotonic() + 30
            while ((TRAIN, INFER), 3) not in agent.batches["b"].dual_round_pending:
                if time.monotonic() > deadline:
                    raise TimeoutError("round-3 send never parked")
                time.sleep(0.05)
            assert not fut.done(), "a parked send must not complete before its peer arrives"
            handler_infer.close()
            with pytest.raises(RuntimeError, match="cleaned up before direction-round executed"):
                fut.result(timeout=30)

        handler_train.close()
        # A new temporary sink reuses this batch ID after the old handlers
        # close. Round 0 must not inherit the previous completion signal.
        new_target = torch.zeros_like(target)
        with ThreadPoolExecutor(max_workers=2) as pool:
            new_train = pool.submit(
                client_train.register_tensors, batch_id="b", tensors=[(master, pair)], role=TRAIN, timeout=60
            )
            new_infer = pool.submit(
                client_infer.register_tensors, batch_id="b", tensors=[(new_target, pair)], role=INFER, timeout=60
            )
            handler_train = new_train.result(timeout=90)
            handler_infer = new_infer.result(timeout=90)
        assert handler_infer.query_transfer_signal(sync_round=0) is False
        master.fill_(42)
        _round_transfer(handler_train, handler_infer, TRAIN, INFER, 0)
        assert torch.equal(new_target, master)
        assert handler_infer.query_transfer_signal(sync_round=0) is True
        handler_train.close()
        handler_infer.close()
        client_train.close()
        client_infer.close()
    finally:
        stop.set()
        thread.join(timeout=30)
        agent.close(destroy=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        # Subgroup cache holds handles of the destroyed world; a later
        # main-process test would reuse them (see test_dual_endpoint_agent).
        _PROCESS_GROUP_CACHE.clear()


def _multi_pair_worker(rank: int, world_size: int, root: str, store_port: int, dist_port: int) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), MASTER_ADDR="localhost", MASTER_PORT=str(dist_port))
    cmd_path, state_path = f"{root}/{rank}_command.lmdb", f"{root}/{rank}_state.lmdb"

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
    try:
        pair_a, pair_b = "pair_a", "pair_b"  # sorted order: a before b
        client_train = TensorBusClient(
            agent_rank=rank, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path
        )
        client_infer = TensorBusClient(
            agent_rank=rank, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path
        )
        # pair_a: cross-rank reshard, rows -> column halves (the ARC2 shape);
        # pair_b: replicated full copy — a different placement, same batch.
        # All four InitPair commands concurrently: the first parks until the
        # rendezvous completes, so sequential same-thread issue deadlocks.
        inits = [
            (client_train, pair_a, TRAIN, INFER, world_size, _MeshSpec((world_size,)), (Shard(0),)),
            (client_infer, pair_a, INFER, TRAIN, world_size, _MeshSpec((1, world_size)), (Replicate(), Shard(1))),
            (client_train, pair_b, TRAIN, INFER, world_size, _MeshSpec((world_size,)), (Replicate(),)),
            (client_infer, pair_b, INFER, TRAIN, world_size, _MeshSpec((world_size,)), (Replicate(),)),
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [
                pool.submit(c.init_pair, p, ln, rn, ws, mesh, plc, timeout=60) for c, p, ln, rn, ws, mesh, plc in inits
            ]
            for f in futs:
                f.result(timeout=90)

        rows, cols = 8, 4
        torch.manual_seed(100)  # identical global tensors on every rank
        full_a1 = torch.randn(rows, cols)
        full_a2 = torch.randn(rows, cols)
        full_b1 = torch.randn(cols, cols)
        full_b2 = torch.randn(cols, cols)

        train_local = [
            # clone(): a chunk of a contiguous tensor is itself contiguous, so
            # .contiguous() would alias the source storage and round r's copy_
            # would corrupt the expectations computed from the full tensors
            full_a1.chunk(world_size, dim=0)[rank].clone(),
            full_a2.chunk(world_size, dim=0)[rank].clone(),
            full_b1.clone(),
            full_b2.clone(),
        ]
        infer_local = [
            torch.zeros(rows, cols // world_size, dtype=torch.bfloat16),  # a1: column half
            torch.zeros(rows, cols // world_size, dtype=torch.bfloat16),  # a2
            torch.zeros(cols, cols),  # b1: full fp32 copy
            torch.zeros(cols, cols),  # b2
        ]

        tensors_train = [
            (train_local[0], pair_a),
            (train_local[1], pair_a),
            (train_local[2], pair_b),
            (train_local[3], pair_b),
        ]
        tensors_infer = [
            (infer_local[0], pair_a),
            (infer_local[1], pair_a),
            (infer_local[2], pair_b),
            (infer_local[3], pair_b),
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_train = pool.submit(
                client_train.register_tensors, batch_id="multi", tensors=tensors_train, role=TRAIN, timeout=60
            )
            fut_infer = pool.submit(
                client_infer.register_tensors, batch_id="multi", tensors=tensors_infer, role=INFER, timeout=60
            )
            handler_train = fut_train.result(timeout=90)
            handler_infer = fut_infer.result(timeout=90)

        def pattern(full: torch.Tensor, rnd: int) -> torch.Tensor:
            # distinct global coordinates per round: round offset + column ramp
            return full + float(rnd + 1) * 100.0 + torch.arange(full.shape[-1])

        for rnd in range(2):
            exp_a1, exp_a2 = pattern(full_a1, rnd), pattern(full_a2, rnd)
            exp_b1, exp_b2 = pattern(full_b1, rnd), pattern(full_b2, rnd)
            train_local[0].copy_(exp_a1.chunk(world_size, dim=0)[rank])
            train_local[1].copy_(exp_a2.chunk(world_size, dim=0)[rank])
            train_local[2].copy_(exp_b1)
            train_local[3].copy_(exp_b2)

            # rank parity flips which role's command arrives first this round
            _round_transfer(handler_train, handler_infer, TRAIN, INFER, rnd, send_first=(rank + rnd) % 2 == 0)

            col = rank % world_size
            want_a1 = exp_a1.to(torch.bfloat16).chunk(world_size, dim=1)[col]
            assert torch.equal(infer_local[0], want_a1), (
                f"rank {rank} round {rnd}: pair_a tensor 0 not delivered: "
                f"got {infer_local[0].tolist()} want {want_a1.tolist()}"
            )
            assert torch.equal(infer_local[1], exp_a2.to(torch.bfloat16).chunk(world_size, dim=1)[col]), (
                f"rank {rank} round {rnd}: pair_a tensor 1 not delivered"
            )
            assert torch.equal(infer_local[2], exp_b1), f"rank {rank} round {rnd}: pair_b tensor 0 not delivered"
            assert torch.equal(infer_local[3], exp_b2), f"rank {rank} round {rnd}: pair_b tensor 1 not delivered"
            assert handler_infer.query_transfer_signal(sync_round=rnd) is True

        handler_train.close()
        handler_infer.close()
        client_train.close()
        client_infer.close()
    finally:
        stop.set()
        thread.join(timeout=30)
        agent.close(destroy=True)
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.timeout(600)
def test_multi_pair_dual_batch_transfers_every_pair(tmp_path):
    """Two placement-different pairs, one batch, two rounds: all tensors land."""
    store_port, dist_port = _free_port(), _free_port()
    try:
        torch.multiprocessing.spawn(
            _multi_pair_worker, args=(2, str(tmp_path), store_port, dist_port), nprocs=2, join=True
        )
    except Exception as e:
        pytest.fail(f"multi-pair dual batch failed: {e}")


def _out_of_order_worker(rank: int, world_size: int, root: str, store_port: int, dist_port: int) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), MASTER_ADDR="localhost", MASTER_PORT=str(dist_port))
    cmd_path, state_path = f"{root}/{rank}_command.lmdb", f"{root}/{rank}_state.lmdb"

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
    try:
        pair_a, pair_b = "zz_pair_b_first", "aa_pair_a_second"  # issue order != name order
        first, second = (pair_a, pair_b) if rank == 0 else (pair_b, pair_a)
        client_train = TensorBusClient(
            agent_rank=rank, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path
        )
        client_infer = TensorBusClient(
            agent_rank=rank, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path
        )
        for pair in (first, second):
            # non-blocking init: commands for both pairs are in flight before
            # either completes, so the two agents' insertion orders differ
            client_train.init_pair(
                pair, TRAIN, INFER, world_size, _MeshSpec((world_size,)), (Shard(0),), blocking=False
            )
            client_infer.init_pair(
                pair, INFER, TRAIN, world_size, _MeshSpec((world_size,)), (Replicate(),), blocking=False
            )

        deadline = time.monotonic() + 60
        while not {pair_a, pair_b} <= set(agent.pairs):
            if time.monotonic() > deadline:
                raise TimeoutError(f"rank {rank}: pairs never completed (pending={sorted(agent.pending_pairs)})")
            time.sleep(0.05)

        # both pairs usable: one dual batch spanning them transfers correctly.
        # Shard(0) over 2 ranks: local halves of global 4 / global 6 tensors.
        t_a = torch.full((2,), float(rank), dtype=torch.float32)  # -> [0,0,1,1]
        t_b = torch.full((3,), float(10 + rank), dtype=torch.float32)  # -> [10,10,10,11,11,11]
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_train = pool.submit(
                client_train.register_tensors,
                batch_id="oo",
                tensors=[(t_a, pair_a), (t_b, pair_b)],
                role=TRAIN,
                timeout=60,
            )
            fut_infer = pool.submit(
                client_infer.register_tensors,
                batch_id="oo",
                tensors=[(torch.zeros(4), pair_a), (torch.zeros(6), pair_b)],
                role=INFER,
                timeout=60,
            )
            handler_train = fut_train.result(timeout=90)
            handler_infer = fut_infer.result(timeout=90)
        _round_transfer(handler_train, handler_infer, TRAIN, INFER, 0)
        # Replicate targets hold the full concatenated Sharded source
        assert torch.equal(
            agent.batches["oo"].pair_role_tensors[(pair_a, INFER)][0], torch.tensor([0.0, 0.0, 1.0, 1.0])
        )
        assert torch.equal(
            agent.batches["oo"].pair_role_tensors[(pair_b, INFER)][0], torch.tensor([10.0] * 3 + [11.0] * 3)
        )
        handler_train.close()
        handler_infer.close()
        client_train.close()
        client_infer.close()
    finally:
        stop.set()
        thread.join(timeout=30)
        agent.close(destroy=True)
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.timeout(300)
def test_pair_init_out_of_order_no_deadlock(tmp_path):
    """Rank-reversed InitPair arrival orders must complete both pairs."""
    store_port, dist_port = _free_port(), _free_port()
    try:
        torch.multiprocessing.spawn(
            _out_of_order_worker, args=(2, str(tmp_path), store_port, dist_port), nprocs=2, join=True
        )
    except Exception as e:
        pytest.fail(f"out-of-order pair init deadlocked or failed: {e}")


@pytest.mark.timeout(300)
def test_request_signal_drives_reactive_recv(tmp_path):
    """The production notification path: the recv is REACTIVE, not concurrent.

    The trainer issues ONLY its blocking send (it parks on the round
    handshake). A watcher — standing in for the engine service — polls
    query_transfer_request for the forward direction and issues the round's
    recv only once the request is published. Before the send: no request.
    After the send parks: the request IS visible, the completion signal is
    NOT, and the send stays parked until the watcher's recv arrives.
    """
    root = str(tmp_path)
    store_port, dist_port = _free_port(), _free_port()
    if dist.is_initialized():  # main-process tests must not stack PGs
        dist.destroy_process_group()
    os.environ.update(RANK="0", WORLD_SIZE="1", MASTER_ADDR="localhost", MASTER_PORT=str(dist_port))
    cmd_path, state_path = f"{root}/command.lmdb", f"{root}/state.lmdb"
    _cleanup_stale(cmd_path, state_path)

    agent = TensorBusAgent(
        rank=0,
        world_size=1,
        store_host="localhost",
        store_port=store_port,
        lmdb_command_queue_path=cmd_path,
        lmdb_state_path=state_path,
        dist_backend="cpu:gloo",
    )
    stop = threading.Event()
    thread = threading.Thread(target=_agent_loop, args=(agent, stop), daemon=True)
    thread.start()
    try:
        pair = "rq"
        client_train = TensorBusClient(agent_rank=0, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path)
        client_infer = TensorBusClient(agent_rank=0, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [
                pool.submit(client_train.init_pair, pair, TRAIN, INFER, 1, _MeshSpec((1,)), (Shard(0),), timeout=60),
                pool.submit(client_infer.init_pair, pair, INFER, TRAIN, 1, _MeshSpec((1,)), (Replicate(),), timeout=60),
            ]
            for fut in futs:
                fut.result(timeout=90)

        master = torch.arange(12, dtype=torch.float32).view(6, 2)
        target = torch.zeros(6, 2, dtype=torch.float32)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [
                pool.submit(
                    client_train.register_tensors, batch_id="b", tensors=[(master, pair)], role=TRAIN, timeout=60
                ),
                pool.submit(
                    client_infer.register_tensors, batch_id="b", tensors=[(target, pair)], role=INFER, timeout=60
                ),
            ]
            handler_train = futs[0].result(timeout=90)
            handler_infer = futs[1].result(timeout=90)

        # Nothing was sent yet: the watcher must NOT see a request.
        assert handler_infer.query_transfer_request((TRAIN, INFER), sync_round=0) is False

        for rnd in range(2):
            master.copy_(torch.full_like(master, float(rnd + 1)) + torch.arange(12, dtype=torch.float32).view(6, 2))
            expected = master.clone()
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(
                    handler_train.transfer, transfer_type="send", role=TRAIN, blocking=True, timeout=60, sync_round=rnd
                )
                deadline = time.monotonic() + 30
                while not handler_infer.query_transfer_request((TRAIN, INFER), sync_round=rnd):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"round {rnd}: the parked send never published its request")
                    time.sleep(0.05)
                assert not fut.done(), "the send must stay parked until the watcher's recv arrives"
                assert handler_infer.query_transfer_signal(sync_round=rnd) is False, (
                    "request is not completion — the round cannot be complete before the recv exists"
                )
                assert handler_infer.query_transfer_request((TRAIN, INFER), sync_round=rnd + 1) is False, (
                    "a later round must not look requested"
                )
                # The reactive side: only NOW does the engine issue the recv.
                handler_infer.transfer(transfer_type="recv", role=INFER, blocking=True, timeout=60, sync_round=rnd)
                fut.result(timeout=30)
            assert torch.equal(target, expected), f"round {rnd}: the reactive recv did not land the round's values"
            assert agent.batches["b"].dual_direction_done[(TRAIN, INFER)] == rnd

        handler_train.close()
        handler_infer.close()
        client_train.close()
        client_infer.close()
    finally:
        stop.set()
        thread.join(timeout=30)
        agent.close(destroy=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        # Subgroup cache holds handles of the destroyed world; a later
        # main-process test would reuse them (see test_dual_endpoint_agent).
        _PROCESS_GROUP_CACHE.clear()


def _opposed_order_worker(rank: int, world_size: int, root: str, store_port: int, dist_port: int) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), MASTER_ADDR="localhost", MASTER_PORT=str(dist_port))
    cmd_path, state_path = f"{root}/{rank}_command.lmdb", f"{root}/{rank}_state.lmdb"

    agent = TensorBusAgent(
        rank=rank,
        world_size=world_size,
        store_host="localhost",
        store_port=store_port,
        lmdb_command_queue_path=cmd_path,
        lmdb_state_path=state_path,
        dist_backend="cpu:gloo",
    )
    if rank == world_size - 1:
        # This rank's completion polling is deferred until long after both
        # pairs became ready everywhere: its FIRST poll then sees the full
        # ready set, where a per-rank sorted() would order by NAME (aa before
        # zz) against the order the other ranks complete in (zz first — it
        # was ready first). The completion log must serialize everyone.
        release = time.monotonic() + 3.0
        orig_poll = agent._poll_pending_pairs

        def deferred_poll():
            if time.monotonic() < release:
                return
            agent._poll_pending_pairs = orig_poll
            orig_poll()

        agent._poll_pending_pairs = deferred_poll
    stop = threading.Event()
    thread = threading.Thread(target=_agent_loop, args=(agent, stop), daemon=True)
    thread.start()
    try:
        # Name order OPPOSES readiness order: zz_* is ready first but sorts
        # last. Mesh shapes (2,3) vs (3,2) create disjoint REAL mesh
        # sub-groups — the only surface where per-rank group-creation order
        # can actually diverge. A 2D mesh needs one placement per mesh dim:
        # Shard(0) over dim 0 ((2,3) → 2 halves; (3,2) → 3 thirds).
        pair_first, pair_second = "zz_ready_first", "aa_ready_second"
        client_train = TensorBusClient(
            agent_rank=rank, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path
        )
        client_infer = TensorBusClient(
            agent_rank=rank, lmdb_command_queue_path=cmd_path, agent_state_lmdb_path=state_path
        )
        for pair, mesh_shape in ((pair_first, (2, 3)), (pair_second, (3, 2))):
            client_train.init_pair(
                pair, TRAIN, INFER, world_size, _MeshSpec(mesh_shape), (Shard(0), Replicate()), blocking=False
            )
            client_infer.init_pair(
                pair, INFER, TRAIN, world_size, _MeshSpec(mesh_shape), (Replicate(), Replicate()), blocking=False
            )
            if rank != world_size - 1:
                time.sleep(0.5)  # pair_first's keys land well before pair_second's

        deadline = time.monotonic() + 90
        while not {pair_first, pair_second} <= set(agent.pairs):
            if time.monotonic() > deadline:
                raise TimeoutError(f"rank {rank}: pairs never completed (pending={sorted(agent.pending_pairs)})")
            time.sleep(0.05)
        # dict insertion order == this rank's completion order
        Path(f"{root}/order_rank{rank}.json").write_text(json.dumps(list(agent.pairs)))

        # Both pairs must be USABLE, through ONE dual batch spanning them.
        # Shard(0) over mesh dim 0: (2,3) → 2 row-halves; (3,2) → 3 row-thirds.
        full_first = torch.arange(18, dtype=torch.float32).view(6, 3)
        full_second = torch.arange(12, 24, dtype=torch.float32).view(6, 2)
        train_local = [full_first.chunk(2, dim=0)[rank // 3].clone(), full_second.chunk(3, dim=0)[rank // 2].clone()]
        infer_local = [torch.zeros(6, 3), torch.zeros(6, 2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [
                pool.submit(
                    client_train.register_tensors,
                    batch_id="opp",
                    tensors=[(train_local[0], pair_first), (train_local[1], pair_second)],
                    role=TRAIN,
                    timeout=90,
                ),
                pool.submit(
                    client_infer.register_tensors,
                    batch_id="opp",
                    tensors=[(infer_local[0], pair_first), (infer_local[1], pair_second)],
                    role=INFER,
                    timeout=90,
                ),
            ]
            handler_train = futs[0].result(timeout=120)
            handler_infer = futs[1].result(timeout=120)
        _round_transfer(handler_train, handler_infer, TRAIN, INFER, 0, send_first=(rank % 2 == 0))
        assert torch.equal(infer_local[0], full_first), f"rank {rank}: (2,3)-mesh pair not delivered bitwise"
        assert torch.equal(infer_local[1], full_second), f"rank {rank}: (3,2)-mesh pair not delivered bitwise"
        handler_train.close()
        handler_infer.close()
        client_train.close()
        client_infer.close()
    finally:
        stop.set()
        thread.join(timeout=30)
        agent.close(destroy=True)
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.timeout(600)
def test_pair_completion_order_identical_across_ranks(tmp_path):
    """Completion order must be one global sequence, identical on every rank.

    Different ranks receive different pairs' InitPair commands at different
    times, one rank only starts completing long after both pairs are ready,
    and name order opposes readiness order — a per-rank sorted(ready-set)
    could complete in opposite orders on different ranks, wiring different
    memberships to the same process-group names. The store-backed
    completion log must serialize every rank into the same sequence.
    """
    store_port, dist_port = _free_port(), _free_port()
    try:
        torch.multiprocessing.spawn(
            _opposed_order_worker, args=(6, str(tmp_path), store_port, dist_port), nprocs=6, join=True
        )
    except Exception as e:
        pytest.fail(f"opposed-order pair completion failed: {e}")

    orders = [json.loads(path.read_text()) for path in sorted(Path(str(tmp_path)).glob("order_rank*.json"))]
    assert len(orders) == 6, f"missing completion-order records: {len(orders)}/6"
    assert all(order == orders[0] for order in orders), f"completion order diverged across ranks: {orders}"
    assert orders[0] == ["zz_ready_first", "aa_ready_second"], (
        f"completion order must follow readiness (zz was ready first), got {orders[0]}"
    )
