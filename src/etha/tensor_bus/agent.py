"""Tensor Bus Agent Process."""

import os
import json
import time
import uuid
import logging
import traceback
from multiprocessing.reduction import ForkingPickler

import lmdb
import torch
import msgspec

try:
    import logfire
except ImportError:
    logfire = None
import posix_ipc
import torch.distributed as dist
from upath import UPath
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Partial, Placement

from etha.comm import (
    bucket_comm,
    get_m2m_map,
    m2m_to_chunks,
    chunk_to_bucket_ops,
    prewarm_broadcast_groups,
)
from etha.comm.ir import Chunk
from etha.kvstore import KVStore, create_store
from etha.pg_utils import get_or_create_process_group
from etha.comm.utils import enumerate_partial_subgroup_ranks

from .utils import setup_cuda_rebuild_patch
from .commands import (
    InitPair,
    Transfer,
    QueryStatus,
    CleanupBatch,
    RegisterTensors,
    command_error_key,
    command_result_key,
)
from .pair_state import PairState
from .batch_state import BatchState
from .command_queue import CommandQueue

logger = logging.getLogger(__name__)

TIME_INTERVAL = 0.001  # 1ms
# Error records are reclaimed once older than any plausible client wait (default timeout 30s).
COMMAND_ERROR_TTL = 30 * 60.0


class _InvalidRegistrationError(ValueError):
    """Registration rejection that is safe to report without stopping the agent."""


def _create_partial_groups(
    mesh_tensor: torch.Tensor,
    partial_reductions: list[tuple[int, str]],
    this_rank: int,
    full_source_ranks: list[int],
    full_source_group: dist.ProcessGroup,
) -> list[tuple[dist.ProcessGroup, str]]:
    """Create NCCL sub-groups for each Partial dim; return groups this rank belongs to.

    ``dist.new_group`` is collective on WORLD, so every WORLD rank must call it in
    the same order — non-members included. Reuses ``full_source_group`` when a
    sub-group spans the entire source side (the common 1D-mesh-single-Partial case),
    avoiding a redundant new_group bootstrap.
    """
    my_groups: list[tuple[dist.ProcessGroup, str]] = []
    full_set = set(full_source_ranks)
    for mesh_dim_idx, reduce_op in partial_reductions:
        for sub_ranks in enumerate_partial_subgroup_ranks(mesh_tensor, mesh_dim_idx):
            if set(sub_ranks) == full_set:
                group = full_source_group
            else:
                group = get_or_create_process_group(sub_ranks)
            if this_rank in sub_ranks:
                my_groups.append((group, reduce_op))
    return my_groups


class _PendingPair:
    """Rendezvous state for a pair whose InitPair handshake is in flight.

    The agent writes its side's keys immediately (``_handle_init_pair``) and
    completes the pair from the main-loop poll once every side's registration
    is visible in the store. Per peer name we track the semaphores of the
    InitPair commands that registered it, so completion (or failure) can
    release exactly those clients.
    """

    def __init__(self, pair_name: str, local_name: str, remote_name: str, expected: dict[str, int]):
        self.pair_name = pair_name
        # Perspective of the first InitPair seen for this pair (split-mode
        # local/remote semantics); both peer names are known from any InitPair.
        self.local_name = local_name
        self.remote_name = remote_name
        self.expected = expected
        self.semaphores: dict[str, list[str | None]] = {}

    def add_semaphore(self, name: str, semaphore_name: str | None) -> None:
        self.semaphores.setdefault(name, []).append(semaphore_name)

    def all_semaphores(self) -> list[str]:
        return [s for names in self.semaphores.values() for s in names if s is not None]


class TensorBusAgent:
    """Tensor Bus Agent."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        store_host: str,
        store_port: int,
        lmdb_command_queue_path: str,
        lmdb_state_path: str,
        store_timeout: float = 3600.0,
        store_backend: str = "tcp",
        store_namespace: str | None = None,
        dist_backend: str | None = None,
    ):
        """Initialize Agent.

        Args:
            rank: Rank in the torch.distributed group
            world_size: Total number of Agents
            store_host: KVStore server host
            store_port: KVStore server port
            lmdb_command_queue_path: Path to CommandQueue LMDB
            lmdb_state_path: Path to State LMDB
            store_timeout: KVStore connection timeout in seconds
            store_backend: KVStore backend ("tcp" or "etcd")
            dist_backend: torch.distributed backend string. Defaults to
                "cuda:nccl,cpu:gloo" on CUDA hosts and "cpu:gloo" otherwise
                (CPU instantiation exists for the gloo test ladder).
        """
        self.rank = rank
        self.world_size = world_size

        # Initialize torch.distributed first (needed for namespace broadcast)
        logger.debug(f"Agent {rank}: Initializing torch.distributed")
        if dist_backend is None:
            dist_backend = "cuda:nccl,cpu:gloo" if torch.cuda.is_available() else "cpu:gloo"
        if "nccl" in dist_backend:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend=dist_backend, rank=rank, world_size=world_size)

        if store_namespace is None:
            # Generate namespace: rank 0 creates UUID, broadcasts to all
            if rank == 0:
                store_namespace = uuid.uuid4().hex[:8]
            else:
                store_namespace = None
            namespace_list = [store_namespace]
            dist.broadcast_object_list(namespace_list, src=0)
            store_namespace = namespace_list[0]

        logger.info(f"Agent {rank}: Using namespace '{store_namespace}'")

        # Initialize KVStore with namespace
        logger.info(f"Agent {rank}: Connecting to {store_backend} store at {store_host}:{store_port}")
        self.store: KVStore = create_store(
            host=store_host, port=store_port, timeout=store_timeout, backend=store_backend, namespace=store_namespace
        )

        # Initialize CommandQueue (for Host communication)
        self.command_queue = CommandQueue(lmdb_command_queue_path)

        # Initialize State LMDB (for Worker verification)
        self.lmdb_state_path = UPath(lmdb_state_path)
        self.state_env = lmdb.open(
            lmdb_state_path,
            max_dbs=2,  # Allow multiple named databases
            map_size=1 << 28,  # 256MB
            subdir=False,
            lock=True,
        )
        self.state_db = self.state_env.open_db(b"pair_state")
        self._command_error_times: dict[str, float] = {}
        logger.info(f"Agent {rank}: State LMDB initialized at {lmdb_state_path}")

        # Write initial heartbeat (for connection validation)
        self._update_heartbeat()
        logger.debug(f"Agent {rank}: Initial heartbeat written")

        self.pairs: dict[str, PairState] = {}
        self.batches: dict[str, BatchState] = {}
        self._batch_generation: dict[str, int] = {}
        # Rendezvous state for pairs whose InitPair handshake is not yet
        # complete. InitPair never blocks (see _handle_init_pair); completion
        # is polled from the main loop so one agent can drive both roles of a
        # colocated pair concurrently.
        self.pending_pairs: dict[str, _PendingPair] = {}
        # Cursor into the store-backed completion log (see _poll_pending_pairs):
        # the index of the last log entry this rank consumed. Every rank
        # consumes the same log, so the sequence of _complete_pair calls — and
        # therefore of process-group creations — is identical on every rank.
        # ``_completion_log_len`` is the leader-only write index (the leader
        # also consumes; the two counters must not be shared or its consume
        # loop would skip the entries it just wrote).
        self._completion_cursor = 0
        self._completion_log_len = 0

        setup_cuda_rebuild_patch()

        logger.info(f"Agent {rank}: Initialized successfully")

    def run(self):
        """Main loop: process commands from Host."""
        logger.info(f"Agent {self.rank}: Starting main loop")

        while True:
            self.step()

    def step(self) -> bool:
        """One loop iteration: heartbeat, one command, pending-pair polling.

        Split out of ``run`` so tests (and future embedders) can drive the
        agent single-stepped. Returns True when a command was processed.
        """
        # Update heartbeat (for connection validation)
        self._update_heartbeat()

        if self.pending_pairs:
            self._poll_pending_pairs()

        msg = self.command_queue.dequeue(block=True, timeout=TIME_INTERVAL)
        handled = msg is not None
        if msg is not None:
            self._handle_command(msg)

        if self.pending_pairs:
            self._poll_pending_pairs()

        return handled

    def _handle_command(self, command):
        """Dispatch command to appropriate handler and handle semaphore release."""
        # Only create observation on rank 0
        if int(os.environ.get("RANK")) == 0 and logfire:
            with logfire.span(f"handle-{type(command).__name__}", input=command):
                self._execute_command(command)
        else:
            self._execute_command(command)

    def _execute_command(self, command):
        """Execute the actual command logic."""
        try:
            complete = True
            match command:
                case InitPair():
                    self._handle_init_pair(command)
                case Transfer():
                    complete = self._handle_transfer(command)
                case QueryStatus():
                    self._handle_query_status(command)
                case RegisterTensors():
                    complete = self._handle_register_tensors(command)
                case CleanupBatch():
                    self._handle_cleanup_batch(command)
                case _:
                    logger.warning(f"Agent {self.rank}: Unknown command type: {type(command)}")
                    return  # Don't release semaphore for unknown commands

            # Release semaphore if specified. Two commands complete
            # asynchronously and release at completion (or recorded failure)
            # instead: InitPair in _poll_pending_pairs, because the handler
            # returns before the rendezvous completes; and the first role's
            # RegisterTensors on a dual batch, in _register_dual_role, because
            # generation only runs once the second role registers.
            if command.semaphore_name and not isinstance(command, InitPair):
                if complete:
                    self._release_semaphore(command.semaphore_name)

        except Exception as e:
            logger.error(f"Agent {self.rank}: Error handling command {type(command)}: {e} {traceback.format_exc()}")
            if command.semaphore_name:
                try:
                    self._record_command_error(command.semaphore_name, e)
                except Exception:
                    logger.error(
                        f"Agent {self.rank}: Failed to record command error for {command.semaphore_name}:"
                        f" {traceback.format_exc()}"
                    )
                self._release_semaphore(command.semaphore_name)
            if isinstance(e, _InvalidRegistrationError):
                return
            raise

    def _handle_init_pair(self, msg: InitPair):
        """Handle InitPair: write this side's rendezvous keys, never block.

        The pair completes from the main-loop poll (``_poll_pending_pairs``)
        once every side's registration is visible in the store. Blocking here
        would deadlock a dual-endpoint agent: the first role's InitPair would
        wait for the second role's keys, which are only written by a command
        queued behind it on this same agent. The command's semaphore is
        released at pair completion (or recorded failure), not on return.

        Steps (write phase):
        1. Write to KVStore this side's registration
        2. Write expected_world_size for this side (idempotent)
        3. Write this side's device mesh and placement info (role-ized keys)
        4. Register the pending pair; completion runs the remaining old steps
        """
        pair_name = msg.pair_name
        local_name = msg.local_name
        expected_local = msg.expected_world_size
        remote_name = msg.remote_name

        logger.info(f"Agent {self.rank}: RegisterPair pair={pair_name}, local={local_name} -> remote={remote_name}")

        if pair_name in self.pairs:
            # Duplicate InitPair for an already-matched pair (e.g. client
            # retry): keys are rewritten idempotently and the caller is
            # released immediately — the pair is already usable.
            existing = self.pairs[pair_name]
            peers = set(existing.role_ranks or ()) or {existing.local_name, existing.remote_name}
            if local_name not in peers or remote_name not in peers:
                raise _InvalidRegistrationError(
                    f"Agent {self.rank}: InitPair for pair '{pair_name}' names '{local_name}'->'{remote_name}' "
                    f"(known: {', '.join(sorted(peers))})"
                )
            known_size = (
                len(existing.role_ranks[local_name])
                if existing.role_ranks and local_name in existing.role_ranks
                else (len(existing.local_ranks) if local_name == existing.local_name else len(existing.remote_ranks))
            )
            if expected_local != known_size:
                raise _InvalidRegistrationError(
                    f"Agent {self.rank}: InitPair for pair '{pair_name}' expected_world_size "
                    f"{expected_local} != {known_size}"
                )
            logger.info(f"Agent {self.rank}: Pair '{pair_name}' already matched; InitPair is idempotent")
            self._write_init_pair_keys(msg)
            if msg.semaphore_name:
                self._release_semaphore(msg.semaphore_name)
            return

        # Step 4: record pending state; the main loop completes the pair
        pending = self.pending_pairs.get(pair_name)
        if pending is not None:
            peers = {pending.local_name, pending.remote_name}
            if local_name not in peers or remote_name not in peers:
                raise _InvalidRegistrationError(
                    f"Agent {self.rank}: InitPair for pair '{pair_name}' names '{local_name}'->'{remote_name}' "
                    f"(known: {pending.local_name}, {pending.remote_name})"
                )
            if local_name in pending.expected and pending.expected[local_name] != expected_local:
                raise _InvalidRegistrationError(
                    f"Agent {self.rank}: InitPair for pair '{pair_name}' expected_world_size "
                    f"{expected_local} != {pending.expected[local_name]}"
                )
        self._write_init_pair_keys(msg)
        if pending is None:
            pending = _PendingPair(
                pair_name=pair_name,
                local_name=local_name,
                remote_name=remote_name,
                expected={local_name: expected_local},
            )
            self.pending_pairs[pair_name] = pending
        else:
            pending.expected.setdefault(local_name, expected_local)
            self.store.set(f"pair_completion:logged:{pair_name}", "")
        pending.add_semaphore(local_name, msg.semaphore_name)
        logger.info(
            f"Agent {self.rank}: Pair '{pair_name}' pending; waiting for store rendezvous "
            f"(peers: {pending.local_name}, {pending.remote_name})"
        )

    def _write_init_pair_keys(self, msg: InitPair):
        """Write one side's rendezvous keys: registration, expected size, mesh."""
        pair_name = msg.pair_name
        local_name = msg.local_name

        # Step 1: Write local registration to KVStore
        local_key = f"pair:{pair_name}/rank:{self.rank}/{local_name}"
        self.store.set(local_key, "1")

        # Step 2: Write expected_world_size (all ranks write the same value, idempotent)
        want = str(msg.expected_world_size)
        self.store.set(f"pair:{pair_name}/{local_name}/rank:{self.rank}/expected_world_size", want)
        expected_key = f"pair:{pair_name}/{local_name}/expected_world_size"
        existing = self.store.get(expected_key)
        if existing not in (None, b"", want.encode()) and existing.decode() != want:
            raise _InvalidRegistrationError(
                f"Agent {self.rank}: InitPair for pair '{pair_name}' expected_world_size "
                f"{want} != {existing.decode()}"
            )
        self.store.set(expected_key, want)

        # Step 3: Write device mesh and placement info to store. The peer name
        # is part of the key: a dual-endpoint agent writes two different meshes
        # (one per role) under the same pair, and a rank-only key would have
        # the second role overwrite the first.
        if msg.mesh_shape_payload is not None and msg.placements_payload is not None:
            mesh_shape_key = f"pair:{pair_name}/rank:{self.rank}/{local_name}/mesh_shape"
            self.store.set_bytes(mesh_shape_key, bytes(msg.mesh_shape_payload))

            placements_key = f"pair:{pair_name}/rank:{self.rank}/{local_name}/placements"
            self.store.set_bytes(placements_key, bytes(msg.placements_payload))

    def _check_side_ready(self, pair_name: str, name: str) -> tuple[int, list[int]] | None:
        """Non-blocking readiness of one side: (expected, ranks) or None."""
        present = []
        sizes = []
        for r in range(self.world_size):
            if self.store.get(f"pair:{pair_name}/rank:{r}/{name}") == b"1":
                present.append(r)
                size_bytes = self.store.get(f"pair:{pair_name}/{name}/rank:{r}/expected_world_size")
                if size_bytes in (None, b""):
                    return None
                sizes.append(int(size_bytes.decode()))
        if not sizes or len(set(sizes)) != 1:
            return None
        expected = sizes[0]
        if len(present) < expected:
            return None
        return expected, sorted(present)[:expected]

    def _poll_pending_pairs(self):
        """Complete pending pairs in one globally-serialized order.

        ``dist.new_group`` names new groups by a per-process counter, so every
        rank must create its groups in the SAME sequence or two ranks wire
        different memberships to the same group name. A per-rank
        ``sorted(ready-set)`` is not enough: readiness is store-derived and
        monotonic, but two ranks polling at different instants can hold
        different ready sets (rank A saw only pair X ready and completed it;
        rank B polled after Y also became ready and, when name order opposes
        readiness order, completes Y first) — the sequences then diverge.
        Completion order is therefore serialized through the store by the
        WORLD leader: it appends newly-ready pairs to a monotonic log
        (``pair_completion:entry:{n}``, append is store-only and never blocks
        inside a collective), and every rank — the leader included — consumes
        entries in index order and completes exactly that sequence.

        Assumes every rank received an InitPair for every pair of its world
        (the registration layout check already requires this); an entry for a
        pair this rank never saw is skipped, matching the previous behavior
        for non-member ranks.
        """
        if not self.pending_pairs:
            return
        if self.rank == 0:
            self._append_ready_pairs_to_log()
        self._consume_completion_log()

    def _pair_ready(self, pending: _PendingPair) -> bool:
        """True when both sides of the pair are fully visible in the store."""
        return all(
            self._check_side_ready(pending.pair_name, name) is not None
            for name in (pending.local_name, pending.remote_name)
        )

    def _append_ready_pairs_to_log(self):
        """Leader-only: append newly-ready pairs to the completion log.

        Appended in sorted order among simultaneously-ready pairs, so the log
        sequence is a deterministic function of the store state. Store writes
        only — this must stay collective-free: the leader may be blocked
        inside a previous entry's ``new_group`` rendezvous, and other ranks
        must still be able to read the entries already written.
        """
        for pair_name in sorted(self.pending_pairs):
            marker = f"pair_completion:logged:{pair_name}"
            if self.store.get(marker) not in (None, b""):
                continue
            if not self._pair_ready(self.pending_pairs[pair_name]):
                continue
            self._completion_log_len += 1
            self.store.set(f"pair_completion:entry:{self._completion_log_len}", pair_name)
            self.store.set(marker, "1")
            logger.info(f"Agent {self.rank}: completion log entry {self._completion_log_len}: pair '{pair_name}'")

    def _consume_completion_log(self):
        """Complete pairs strictly in completion-log order on every rank."""
        while True:
            raw = self.store.get(f"pair_completion:entry:{self._completion_cursor + 1}")
            if raw is None:
                return
            self._completion_cursor += 1
            pair_name = raw.decode()
            pending = self.pending_pairs.get(pair_name)
            if pending is None:
                # This rank never received the pair's InitPair (non-member);
                # it creates no groups for it — same as before the log.
                #
                # The skip cannot drop a merely-QUEUED InitPair: the keys the
                # leader's readiness check reads are written only by THIS
                # rank's agent, inside _handle_init_pair, strictly before its
                # pending insertion (same thread, no gap that yields). So
                # keys-visible ⇒ this rank already processed the command ⇒
                # pending exists by the time this loop can consume the entry
                # (the consumer runs in that same thread, only after the
                # handler returned). A rank with the command still queued has
                # no keys in the store, so the pair is not ready and no entry
                # names it yet.
                continue
            try:
                self._complete_pair(pending)
            except Exception as e:
                logger.exception(f"Agent {self.rank}: Pair '{pair_name}' completion failed")
                for semaphore_name in pending.all_semaphores():
                    try:
                        self._record_command_error(semaphore_name, e)
                    except Exception:
                        logger.exception(f"Agent {self.rank}: Failed to record InitPair error")
                    self._release_semaphore(semaphore_name)
                pending.semaphores.clear()
                # Keep pending so a corrected InitPair retry can attach; the
                # retry clears the log marker and is re-queued.
            else:
                for semaphore_name in pending.all_semaphores():
                    self._release_semaphore(semaphore_name)
                self.pending_pairs.pop(pair_name, None)

    def _complete_pair(self, pending: _PendingPair) -> None:
        """Complete a pair named by the ordered ready-pair log."""
        pair_name = pending.pair_name
        sides: dict[str, list[int]] = {}
        for name in (pending.local_name, pending.remote_name):
            ready = self._check_side_ready(pair_name, name)
            assert ready is not None
            sides[name] = ready[1]
        peers_key = f"pair:{pair_name}/canonical_peers"
        payload = json.dumps(sorted((pending.local_name, pending.remote_name)))
        publisher = min(rank for ranks in sides.values() for rank in ranks)
        if self.rank == publisher:
            self.store.set(peers_key, payload)
        got = self.store.wait_for_key(peers_key, timeout=60)
        if json.loads(got) != json.loads(payload):
            raise _InvalidRegistrationError(
                f"Agent {self.rank}: Pair '{pair_name}' peer names {payload} != {got.decode()}"
            )

        # Canonical ordering so all ranks call collectives in the same order;
        # the first-registered local/remote perspective is kept for the split
        # (single-role) fields.
        first_name, second_name = sorted(sides)
        local_name, remote_name = pending.local_name, pending.remote_name
        local_ranks, remote_ranks = sides[local_name], sides[remote_name]
        first_ranks, second_ranks = sides[first_name], sides[second_name]

        logger.info(
            f"Agent {self.rank}: Pair '{pair_name}' rendezvous complete: "
            f"{local_name}={local_ranks}, {remote_name}={remote_ranks}"
        )

        # Collect device mesh and placement info from all ranks (per side)
        local_mesh_info = self._collect_mesh_placement_info(pair_name, local_ranks, local_name)
        remote_mesh_info = self._collect_mesh_placement_info(pair_name, remote_ranks, remote_name)

        # Validate mesh/placement consistency per side
        if local_mesh_info:
            self._validate_mesh_placement_consistency(local_mesh_info)
        if remote_mesh_info:
            self._validate_mesh_placement_consistency(remote_mesh_info)

        dual_endpoint = set(first_ranks) == set(second_ranks)
        if dual_endpoint and not (local_mesh_info and remote_mesh_info):
            raise ValueError(
                f"Agent {self.rank}: Dual-endpoint pair '{pair_name}' requires mesh/placement "
                f"payloads from both roles (no-mesh fallback is single-role only)"
            )

        pair_group = get_or_create_process_group(local_ranks + remote_ranks)

        # Generate P2P maps if validation passed. All ranks take the same
        # branch: the collected info is store-derived and identical everywhere.
        m2m_map_send = None
        m2m_map_recv = None
        m2m_first_to_second = None
        m2m_second_to_first = None
        first_mesh_tensor = None
        second_mesh_tensor = None

        local_is_first = local_name < remote_name

        def _order(loc, rem):
            return (loc, rem) if local_is_first else (rem, loc)

        if local_mesh_info and remote_mesh_info:
            local_mesh_shape, local_placements = local_mesh_info[0]
            local_mesh_tensor = torch.arange(
                local_ranks[0], local_ranks[0] + int(torch.prod(torch.tensor(local_mesh_shape)).item())
            ).view(local_mesh_shape)
            remote_mesh_shape, remote_placements = remote_mesh_info[0]
            remote_mesh_tensor = torch.arange(
                remote_ranks[0], remote_ranks[0] + int(torch.prod(torch.tensor(remote_mesh_shape)).item())
            ).view(remote_mesh_shape)
            logger.info(f"Agent {self.rank}: Local mesh: {local_mesh_tensor} with placements: {local_placements}")
            logger.info(f"Agent {self.rank}: Remote mesh: {remote_mesh_tensor} with placements: {remote_placements}")

            first_mesh_tensor, second_mesh_tensor = _order(local_mesh_tensor, remote_mesh_tensor)
            first_mesh = DeviceMesh("cpu", first_mesh_tensor)
            second_mesh = DeviceMesh("cpu", second_mesh_tensor)
            first_placements, second_placements = _order(local_placements, remote_placements)
            logger.info(f"Agent {self.rank}: Generating M2M maps for pair '{pair_name}'")

            # Partial is supported only on *source* — skip the direction whose
            # target side has Partial. Branch is identical on every rank.
            first_has_partial = any(isinstance(p, Partial) for p in first_placements)
            second_has_partial = any(isinstance(p, Partial) for p in second_placements)

            if not second_has_partial:
                m2m_first_to_second = get_m2m_map(
                    source_mesh=first_mesh,
                    source_placements=first_placements,
                    target_mesh=second_mesh,
                    target_placements=second_placements,
                    group=pair_group,
                    device="cpu",
                )
            else:
                logger.info(
                    f"Agent {self.rank}: Skipping first->second M2M map for pair '{pair_name}' "
                    f"(second side has Partial placement; cross-PG Partial target is not supported)"
                )

            if not first_has_partial:
                m2m_second_to_first = get_m2m_map(
                    source_mesh=second_mesh,
                    source_placements=second_placements,
                    target_mesh=first_mesh,
                    target_placements=first_placements,
                    group=pair_group,
                    device="cpu",
                )
            else:
                logger.info(
                    f"Agent {self.rank}: Skipping second->first M2M map for pair '{pair_name}' "
                    f"(first side has Partial placement; cross-PG Partial target is not supported)"
                )

            # Send is "local -> remote".
            m2m_map_send, m2m_map_recv = _order(m2m_first_to_second, m2m_second_to_first)
            logger.info(
                f"Agent {self.rank}: Generated P2P maps for pair '{pair_name}'. "
                f"m2m_map_send: {m2m_map_send} m2m_map_recv: {m2m_map_recv}"
            )

            for mesh in (first_mesh, second_mesh):
                mesh_ranks = mesh.mesh.flatten().tolist()
                if self.rank in mesh_ranks:
                    for pg in mesh.get_all_groups():
                        # A mesh dim covering the whole WORLD aliases the
                        # default group (and singleton dims hand back None) —
                        # destroying either would tear down the agent itself.
                        if pg is None or pg is dist.group.WORLD:
                            continue
                        dist.destroy_process_group(pg)
            logger.debug(f"Agent {self.rank}: Destroyed DeviceMesh process groups for pair '{pair_name}'")
        else:
            logger.info(f"Agent {self.rank}: Skipping P2P map generation - missing or inconsistent mesh/placement info")

        # Side groups (NCCL groups for send/recv side), created in canonical
        # (first, second) order so non-member ranks call new_group in the same
        # sequence — new_group is WORLD-collective.
        first_group = get_or_create_process_group(first_ranks)
        second_group = get_or_create_process_group(second_ranks)
        local_group, remote_group = _order(first_group, second_group)

        # Both directions' Partial sub-groups are created in deterministic
        # order for the same reason; per-side groups for the dual view.
        first_partial_groups: list[tuple[dist.ProcessGroup, str]] | None = None
        second_partial_groups: list[tuple[dist.ProcessGroup, str]] | None = None
        local_partial_groups: list[tuple[dist.ProcessGroup, str]] | None = None
        if local_mesh_info and remote_mesh_info:
            partial_red_1 = m2m_first_to_second.source_partial_reductions if m2m_first_to_second else []
            partial_red_2 = m2m_second_to_first.source_partial_reductions if m2m_second_to_first else []
            mesh_1_partial_groups = _create_partial_groups(
                first_mesh_tensor, partial_red_1, self.rank, first_ranks, first_group
            )
            mesh_2_partial_groups = _create_partial_groups(
                second_mesh_tensor, partial_red_2, self.rank, second_ranks, second_group
            )

            first_partial_groups, second_partial_groups = mesh_1_partial_groups, mesh_2_partial_groups
            local_partial_groups, _ = _order(mesh_1_partial_groups, mesh_2_partial_groups)
            if local_partial_groups:
                logger.info(
                    f"Agent {self.rank}: Created {len(local_partial_groups)} source Partial sub-group(s) "
                    f"for pair '{pair_name}'"
                )

        state = PairState(
            pair_name=pair_name,
            local_name=local_name,
            local_ranks=local_ranks,
            remote_name=remote_name,
            remote_ranks=remote_ranks,
            pair_size=len(first_ranks) + len(second_ranks),
            local_group=local_group,
            pair_group=pair_group,
            local_is_first=local_is_first,
            m2m_send=m2m_map_send,
            m2m_recv=m2m_map_recv,
            source_partial_groups=local_partial_groups,
            dual_endpoint=dual_endpoint,
            role_ranks={first_name: first_ranks, second_name: second_ranks},
            role_groups={first_name: first_group, second_name: second_group},
            role_mesh_shapes={
                first_name: tuple(first_mesh_tensor.shape) if first_mesh_tensor is not None else (),
                second_name: tuple(second_mesh_tensor.shape) if second_mesh_tensor is not None else (),
            },
            m2m_by_source_role={first_name: m2m_first_to_second, second_name: m2m_second_to_first},
            partial_by_source_role={
                first_name: first_partial_groups or None,
                second_name: second_partial_groups or None,
            },
        )
        self.pairs[pair_name] = state
        if dual_endpoint:
            logger.info(f"Agent {self.rank}: Pair '{pair_name}' is dual-endpoint (both roles on ranks {first_ranks})")

        # Write PairState to State LMDB (for Worker verification)
        state_key = f"pair:{pair_name}/state:match".encode()  # LMDB key
        state_bytes = msgspec.msgpack.encode("matched")
        with self.state_env.begin(write=True, db=self.state_db) as txn:
            txn.put(state_key, state_bytes)

        logger.info(
            f"Agent {self.rank}: Pair '{pair_name}' matched! "
            f"Local '{local_name}': {local_ranks}, Remote '{remote_name}': {remote_ranks}"
        )

    def _handle_transfer(self, msg: Transfer) -> bool:
        """Handle Transfer command for batch tensor transfer.

        Returns True when the command is finished (semaphore releasable);
        False when a dual direction-round parks waiting for the peer role.
        """
        batch_id = msg.batch_id
        transfer_type = msg.transfer_type
        logger.info(f"Agent {self.rank}: Handling transfer for batch '{batch_id}' ({transfer_type})")

        if batch_id not in self.batches:
            raise _InvalidRegistrationError(f"Transfer for unknown batch: {batch_id}")

        batch_state = self.batches[batch_id]
        if msg.generation and batch_state.generation and msg.generation != batch_state.generation:
            raise _InvalidRegistrationError(
                f"Transfer for stale generation {msg.generation} of batch {batch_id} (live {batch_state.generation})"
            )

        if batch_state.dual:
            return self._execute_dual_transfer(msg, batch_state)

        # Set transfer_signal to notify receiver that sender is ready (before barrier)
        transfer_signal_key = f"batch:{batch_id}/state:transfer_signal"
        if transfer_type == "send":
            self._leader_set(transfer_signal_key, "1", batch_state)
            logger.info(
                f"Agent {self.rank}: set key {self.store._prefixed(transfer_signal_key, component='global')} value 1"
            )
        # Synchronize all ranks in batch
        dist.barrier(batch_state.batch_group)
        logger.debug(f"Agent {self.rank}: Batch {batch_id}: All ranks synchronized")

        transfer_started = time.perf_counter()

        # Execute transfer using the flattened buckets
        if batch_state.send_buckets or batch_state.recv_buckets:
            if transfer_type == "send":
                buckets = batch_state.send_buckets
            else:
                buckets = batch_state.recv_buckets

            if buckets:
                logger.info(
                    f"Agent {self.rank}: Batch {batch_id}: Executing bucketized transfer with {len(buckets)} buckets"
                )
                bucket_comm(buckets=buckets)
            else:
                # The other direction produced chunks, but this one is empty —
                # init_pair skipped it because the target side has a Partial
                # placement (cross-PG Partial target is unsupported).
                raise RuntimeError(
                    f"Batch {batch_id}: no {transfer_type} chunks. The pair's "
                    f"{transfer_type} direction has a Partial target, which is "
                    f"not supported. Partial is only valid as a source placement."
                )
        else:
            # Fall back to simple send/recv without P2P optimization
            logger.info(f"Agent {self.rank}: Batch {batch_id}: Using simple send/recv transfer (no P2P map available)")
            for pair_name in batch_state.pair_names:
                pair_state = self.pairs[pair_name]
                for i, tensor in enumerate(batch_state.pair_tensors[pair_name]):
                    logger.debug(
                        f"Agent {self.rank}: Batch {batch_id}: Transferring tensor {i} shape: {tensor.shape} for pair '{pair_name}' using simple send/recv"
                    )
                    if transfer_type == "send":
                        torch.distributed.send(tensor, pair_state.remote_ranks[pair_state.local_ranks.index(self.rank)])
                    elif transfer_type == "recv":
                        torch.distributed.recv(tensor, pair_state.remote_ranks[pair_state.local_ranks.index(self.rank)])
                    logger.debug(f"Agent {self.rank}: Batch {batch_id}: Transfered tensor {i}")

        transfer_time_ms = (time.perf_counter() - transfer_started) * 1000

        dist.barrier(batch_state.batch_group)
        if transfer_type == "recv":
            self._leader_set(transfer_signal_key, "0", batch_state)
        logger.info(f"Agent {self.rank}: Batch {batch_id}: Transfer complete in {transfer_time_ms:.2f} ms wall time")
        return True

    def _execute_dual_transfer(self, msg: Transfer, batch_state: BatchState) -> bool:
        """Execute one direction-round of a dual-endpoint (colocated) batch.

        One agent holds both roles' tensors, so both roles' Transfer commands
        for a direction arrive on this same agent: ``send`` from role R and
        ``recv`` from the other role map to the same (R → other) direction,
        identified further by ``sync_round``.

        Handshake: a direction-round executes only after BOTH roles' commands
        for that round arrived — the source side's command is source-ready,
        the destination side's command is dest-ready (its engine is done
        reading the previous weights). The first command therefore PARKS (its
        semaphore is deferred) instead of executing alone; whoever arrives
        second triggers execution. A command for an already-executed or older
        round is acknowledged without re-execution, so a same batch supports
        unbounded sync rounds and duplicates/stale re-issues cannot roll
        weights back.

        Completion is published as a VERSIONED signal
        (``batch:{id}/state:transfer_signal_round`` = round id, monotonic):
        a watcher that polls ``query_transfer_signal(sync_round=r)`` cannot
        lose a pulse the way the legacy boolean flag's 1→0 reset could.

        A REQUEST signal is published BEFORE the first command parks
        (``transfer_request_round:{src}->{dst}``, monotonic per direction):
        the completion signal can only say a round finished, but a reactive
        peer — an engine service that must quiesce and only then issue the
        matching recv — needs to learn the round was asked for. Driving the
        recv off the completion signal would deadlock: the round completes
        only after the recv arrives.

        Returns True when this command is finished (its semaphore may
        release); False when it parks waiting for the peer role's command of
        the same round.
        """
        batch_id = msg.batch_id
        if msg.role is None:
            raise _InvalidRegistrationError(
                f"Batch {batch_id}: dual-endpoint transfer requires the issuing role (Transfer.role); got None"
            )
        first_pair = self.pairs[batch_state.pair_names[0]]
        if msg.role not in first_pair.role_ranks:
            raise _InvalidRegistrationError(
                f"Batch {batch_id}: role '{msg.role}' is not a peer of pair '{first_pair.pair_name}'"
            )
        other_roles = [name for name in first_pair.role_ranks if name != msg.role]
        if not other_roles:
            raise _InvalidRegistrationError(
                f"Batch {batch_id}: role '{msg.role}' is not a peer of pair '{first_pair.pair_name}'"
            )
        other_role = other_roles[0]

        if msg.transfer_type == "send":
            src_role, dst_role = msg.role, other_role
        else:
            src_role, dst_role = other_role, msg.role
        direction = (src_role, dst_role)
        if msg.sync_round < 0:
            raise _InvalidRegistrationError(f"Batch {batch_id}: sync_round must be >= 0, got {msg.sync_round}")
        if batch_state.dual_direction is None:
            key = f"batch:{batch_id}/g{batch_state.generation}/state:dual_direction"
            if self.rank == batch_state.local_leader and self.store.get(key, component="global") is None:
                self.store.set(key, json.dumps(direction), component="global")
            batch_state.dual_direction = tuple(json.loads(self.store.wait_for_key(key, timeout=60, component="global")))
        if direction != batch_state.dual_direction:
            raise _InvalidRegistrationError(
                f"Batch {batch_id}: dual-endpoint batch supports one direction only "
                f"({batch_state.dual_direction}), got {direction}"
            )

        done_round = batch_state.dual_direction_done.get(direction, -1)
        if msg.sync_round <= done_round:
            logger.info(
                f"Agent {self.rank}: Batch {batch_id}: direction {src_role}->{dst_role} round {msg.sync_round} "
                f"already executed (done={done_round}); acknowledging without re-execution"
            )
            pending = batch_state.dual_round_pending.pop((direction, msg.sync_round), None)
            if pending:
                for parked in pending["semaphores"]:
                    self._release_semaphore(parked)
            return True

        pending = batch_state.dual_round_pending.setdefault(
            (direction, msg.sync_round), {"roles": set(), "semaphores": []}
        )
        if not pending["roles"]:
            # First command for this direction-round on this rank: publish
            # the request BEFORE parking (see docstring) — after the park it
            # is too late for a watcher to react, and the completion signal
            # is definitionally not out yet.
            self._publish_transfer_request(batch_state, direction, msg.sync_round)
        pending["roles"].add(msg.role)
        if msg.semaphore_name:
            pending["semaphores"].append(msg.semaphore_name)
        if len(pending["roles"]) < 2:
            logger.info(
                f"Agent {self.rank}: Batch {batch_id}: direction {src_role}->{dst_role} round {msg.sync_round} "
                f"parked (have {sorted(pending['roles'])}); waiting for the peer role's command"
            )
            return False

        def _fail_round(err: BaseException) -> None:
            parked_round = batch_state.dual_round_pending.pop((direction, msg.sync_round), None)
            if parked_round:
                for parked in parked_round["semaphores"]:
                    if parked != msg.semaphore_name:
                        self._record_command_error(parked, err)
                        self._release_semaphore(parked)
            if self.rank == batch_state.local_leader:
                prev = batch_state.dual_direction_done.get(direction, -1)
                src_role, dst_role = direction
                self.store.set(
                    f"batch:{batch_id}/state:transfer_request_round:{src_role}->{dst_role}",
                    str(prev),
                    component="global",
                )
            raise err

        if direction not in batch_state.dual_direction_buckets:
            _fail_round(
                _InvalidRegistrationError(
                    f"Batch {batch_id}: no buckets for direction {src_role}->{dst_role}. "
                    f"Both roles must register before transferring."
                )
            )
        buckets = batch_state.dual_direction_buckets[direction]
        if buckets is None:
            _fail_round(
                _InvalidRegistrationError(
                    f"Batch {batch_id}: direction {src_role}->{dst_role} has a Partial target, which is not supported"
                )
            )

        seen_rounds = [None] * dist.get_world_size(batch_state.batch_group)
        dist.all_gather_object(seen_rounds, msg.sync_round, group=batch_state.batch_group)
        if any(round_id != msg.sync_round for round_id in seen_rounds):
            _fail_round(
                _InvalidRegistrationError(
                    f"Batch {batch_id}: mixed sync_round {seen_rounds} for {src_role}->{dst_role}"
                )
            )
        dist.barrier(batch_state.batch_group)
        logger.info(
            f"Agent {self.rank}: Batch {batch_id}: Executing dual transfer {src_role}->{dst_role} "
            f"round {msg.sync_round} with {len(buckets)} buckets"
        )
        bucket_comm(buckets=buckets, sequential=True)
        batch_state.dual_direction_done[direction] = msg.sync_round

        # Publish the versioned completion signal before any parked semaphore
        # releases: when a recv client's blocking call returns, the round is
        # already queryable, so derived tensors may refresh immediately.
        dist.barrier(batch_state.batch_group)
        if self.rank == batch_state.local_leader:
            self.store.set(f"batch:{batch_id}/state:transfer_signal_round", str(msg.sync_round), component="global")
        dist.barrier(batch_state.batch_group)
        for parked in pending["semaphores"]:
            self._release_semaphore(parked)
        del batch_state.dual_round_pending[(direction, msg.sync_round)]
        logger.info(
            f"Agent {self.rank}: Batch {batch_id}: Dual transfer {src_role}->{dst_role} round {msg.sync_round} complete"
        )
        return True

    def _publish_transfer_request(self, batch_state: BatchState, direction: tuple[str, str], sync_round: int) -> None:
        """Publish a direction's REQUEST round — monotonic, never reset.

        Written when the direction-round's FIRST command arrives (before it
        parks). ``transfer_signal_round`` (completion) can only say a round
        FINISHED; a reactive peer — quiesce on request, issue the matching
        recv, then wait for completion — must learn the round was REQUESTED.
        The request is a separate key per direction with the same
        no-pulse-loss property on the start edge that the completion signal
        has on the finish edge.
        """
        if self.rank != batch_state.local_leader:
            return
        src_role, dst_role = direction
        key = f"batch:{batch_state.batch_id}/state:transfer_request_round:{src_role}->{dst_role}"
        raw = self.store.get(key, component="global")
        current = int(raw.decode()) if raw is not None else -1
        if sync_round > current:
            self.store.set(key, str(sync_round), component="global")
            logger.info(
                f"Agent {self.rank}: Batch {batch_state.batch_id}: request round {sync_round} "
                f"published for {src_role}->{dst_role} (was {current})"
            )

    def _handle_query_status(self, msg: QueryStatus):
        batch_id = msg.batch_id
        state_name = msg.state_name  # e.g. "transfer_signal"
        store_state_key = f"batch:{batch_id}/state:{state_name}"
        statedb_key = f"batch:{batch_id}/state:{state_name}".encode()  # LMDB key

        if batch_id not in self.batches:
            logger.error(f"Agent {self.rank}: QueryStatus for unknown batch: {batch_id}")
            return

        batch = self.batches[batch_id]
        if msg.generation and batch.generation and msg.generation != batch.generation:
            state = False if state_name == "transfer_signal" else -1
            if msg.semaphore_name:
                self._write_command_result(msg.semaphore_name, state)
            else:
                with self.state_env.begin(write=True, db=self.state_db) as txn:
                    txn.put(statedb_key, msgspec.msgpack.encode(state))
            return

        if state_name == "transfer_signal":
            raw_value = self.store.get(store_state_key, component="global")
            state = raw_value == b"1"
            logger.info(f"Agent {self.rank}: Query {state_name} batch={batch_id}: raw={raw_value}, state={state}")
        elif state_name == "transfer_signal_round":
            # Versioned dual-endpoint completion: the store holds the last
            # completed round id (monotonic), published after execution.
            raw_value = self.store.get(store_state_key, component="global")
            state = int(raw_value.decode()) if raw_value is not None else -1
            logger.info(f"Agent {self.rank}: Query {state_name} batch={batch_id}: raw={raw_value}, state={state}")
        elif state_name.startswith("transfer_request_round"):
            # Dual-endpoint REQUEST signal, per direction: the last round id
            # whose first command arrived — published before that command
            # parks, monotonic, never reset. Drives reactive recv issuers
            # (quiesce, then issue the matching round's recv).
            raw_value = self.store.get(store_state_key, component="global")
            state = int(raw_value.decode()) if raw_value is not None else -1
            logger.info(f"Agent {self.rank}: Query {state_name} batch={batch_id}: raw={raw_value}, state={state}")
        else:
            logger.error(f"Agent {self.rank}: Invalid state name: {state_name}")
            return

        if msg.semaphore_name:
            self._write_command_result(msg.semaphore_name, state)
        else:
            with self.state_env.begin(write=True, db=self.state_db) as txn:
                txn.put(statedb_key, msgspec.msgpack.encode(state))

    def _handle_cleanup_batch(self, msg: CleanupBatch):
        """Handle CleanupBatch command to free batch state resources."""
        batch_id = msg.batch_id

        if batch_id in self.batches:
            live = self.batches[batch_id]
            if msg.generation and live.generation and msg.generation != live.generation:
                logger.info(
                    f"Agent {self.rank}: ignoring stale CleanupBatch for {batch_id} "
                    f"gen {msg.generation} (live {live.generation})"
                )
                return
            # A dual batch that never reached its second role still parks the
            # first role's RegisterTensors semaphore — and a direction-round
            # whose peer command never arrived parks transfer semaphores.
            # Both release HERE so a failed batch cannot hang its clients, but
            # FAIL-CLOSED: the error is recorded first, so the parked blocking
            # call raises instead of returning as if it had succeeded.
            for parked in self.batches[batch_id].pending_register_semaphores:
                self._record_command_error(
                    parked, RuntimeError(f"Batch {batch_id}: cleaned up before registration completed")
                )
                self._release_semaphore(parked)
            for pending in self.batches[batch_id].dual_round_pending.values():
                for parked in pending["semaphores"]:
                    self._record_command_error(
                        parked, RuntimeError(f"Batch {batch_id}: cleaned up before direction-round executed")
                    )
                    self._release_semaphore(parked)
            batch = self.batches[batch_id]
            if batch.dual and self.rank == batch.local_leader:
                self._reset_dual_store_signals(batch)
            del self.batches[batch_id]
            logger.info(f"Agent {self.rank}: Cleaned up batch {batch_id}")
        else:
            logger.warning(f"Agent {self.rank}: Cleanup requested for unknown batch {batch_id}")

    def _reset_dual_store_signals(self, batch: BatchState) -> None:
        """Drop store-backed round signals so a reused batch id cannot inherit them.

        TCPStore cannot delete keys, so write ``-1`` (the same missing-round
        sentinel the query path uses). Do not clear ``dual_direction``: a stale
        empty value would make ``wait_for_key`` return immediately on reuse.
        Same-direction reuse keeps the pin; a opposite-direction reuse still
        fails closed.
        """
        batch_id = batch.batch_id
        self.store.set(f"batch:{batch_id}/state:transfer_signal_round", "-1", component="global")
        directions = set(batch.dual_direction_buckets)
        if batch.dual_direction:
            directions.add(batch.dual_direction)
        for src_role, dst_role in directions:
            self.store.set(
                f"batch:{batch_id}/state:transfer_request_round:{src_role}->{dst_role}",
                "-1",
                component="global",
            )

    def _handle_register_tensors(self, msg: RegisterTensors):
        """Handle RegisterTensors command for batch tensor registration.

        Split pairs (one role per agent) keep the legacy path: one BatchState
        per registration, chunks generated immediately with the rank0 dtype
        exchange. Dual-endpoint (colocated) pairs merge both roles'
        registrations into one batch — never overwriting — and generate
        chunks only once both roles are present: the agent then holds both
        roles' dtypes locally (no exchange needed; the rank0 P2P exchange
        would be a self-send on colocated ranks).

        Returns True when the registration completed (batch immediately
        usable). Returns False when a dual batch still waits for its other
        role — the command's semaphore is then released at completion by
        ``_register_dual_role`` (or ``_handle_cleanup_batch``), not here.
        """
        batch_id = msg.batch_id
        tensors = msg.tensors  # list[tuple[str, memoryview]]
        bucket_size = msg.bucket_size
        role = msg.role

        logger.info(f"Agent {self.rank}: Starting batch {batch_id}: {len(tensors)} tensors")

        # Group tensors by pair_name
        grouped: dict[str, list[memoryview]] = {}
        for pair_name, tensor_payload in tensors:
            if pair_name not in grouped:
                grouped[pair_name] = []
            grouped[pair_name].append(tensor_payload)

        pair_names = sorted(grouped)

        # A batch is dual-endpoint when any (hence every) pair hosts both
        # roles on the agent ranks; mixing dual and split pairs in one batch
        # is rejected — the two paths differ in merge and chunk semantics.
        pair_dual = {
            name: (self.pairs[name].dual_endpoint if self.pairs.get(name) is not None else False) for name in pair_names
        }
        batch_dual = any(pair_dual.values())
        mix_ok = (not batch_dual) or all(pair_dual.values())
        role_ok = True
        if batch_dual:
            if role is None:
                role_ok = False
            else:
                for name in pair_names:
                    pair = self.pairs.get(name)
                    if pair is None or role not in pair.role_ranks:
                        role_ok = False
                        break
        elif role is not None:
            role_ok = False
        local_ok = mix_ok and role_ok
        role_dup = False
        if batch_dual and role is not None:
            existing_batch = self.batches.get(batch_id)
            if existing_batch is not None:
                role_dup = any((name, role) in existing_batch.pair_role_tensors for name in pair_names)

        pair_layout = []
        local_memberships = set()
        for name in pair_names:
            pair = self.pairs.get(name)
            membership = None
            if pair is not None:
                local_ranks = tuple(sorted(pair.local_ranks))
                remote_ranks = tuple(sorted(pair.remote_ranks))
                local_memberships.add((local_ranks, remote_ranks))
                membership = tuple(sorted((local_ranks, remote_ranks)))
            pair_layout.append((name, len(grouped[name]), membership))
        local_membership_valid = len(local_memberships) == 1
        layout = (batch_id, bucket_size, tuple(pair_layout), local_membership_valid, role, local_ok, role_dup)
        layouts = [None] * self.world_size
        dist.all_gather_object(layouts, layout, group=dist.group.WORLD)
        if any(other is not None and len(other) >= 7 and other[6] for other in layouts):
            err = _InvalidRegistrationError(
                f"Batch {batch_id}: role '{role}' already registered; dual-endpoint batches merge roles and never overwrite"
            )
            existing = self.batches.get(batch_id)
            if existing is not None and existing.pending_register_semaphores:
                self._abort_dual_registration(existing, err)
            raise err
        if any(other is None or (len(other) >= 6 and not other[5]) for other in layouts):
            if batch_dual and role is None:
                err = _InvalidRegistrationError(
                    f"Batch {batch_id} registers a dual-endpoint pair; RegisterTensors.role is required"
                )
            elif not batch_dual and role is not None:
                err = _InvalidRegistrationError(
                    f"Batch {batch_id} has no dual-endpoint pair; RegisterTensors.role must be None"
                )
            else:
                err = _InvalidRegistrationError(f"Batch {batch_id}: invalid registration: {layouts}")
            existing = self.batches.get(batch_id)
            if existing is not None and existing.pending_register_semaphores:
                self._abort_dual_registration(existing, err)
            raise err
        memberships = {membership for _, _, membership in pair_layout}
        if not pair_names or None in memberships or len(memberships) != 1 or not local_membership_valid:
            raise _InvalidRegistrationError(f"Inconsistent or invalid RegisterTensors layout across ranks: {layouts}")

        if batch_dual:
            # Role commands may arrive in either order; the layout signature
            # excludes role and must agree across both gathers and all ranks.
            batch_state = self.batches.get(batch_id)
            if batch_state is None:
                batch_state = BatchState(
                    batch_id=batch_id,
                    pair_names=pair_names,
                    bucket_size=bucket_size,
                    dual=True,
                    generation=self._next_batch_generation(batch_id),
                )
                self.batches[batch_id] = batch_state
            signature = layout[:4]
            if any(other is None or other[:4] != signature for other in layouts):
                err = _InvalidRegistrationError(f"Batch {batch_id}: inconsistent dual layout across ranks: {layouts}")
                self._abort_dual_registration(batch_state, err)
                raise err
            if batch_state.dual_layout_signature is not None and batch_state.dual_layout_signature != signature:
                err = _InvalidRegistrationError(f"Batch {batch_id}: dual role layout changed between registrations")
                self._abort_dual_registration(batch_state, err)
                raise err
            batch_state.dual_layout_signature = signature
            return self._register_dual_role(batch_state, grouped, role, msg.semaphore_name)

        if any(other != layout for other in layouts):
            raise _InvalidRegistrationError(f"Inconsistent or invalid RegisterTensors layout across ranks: {layouts}")

        batch_state = BatchState(
            batch_id=batch_id,
            pair_names=pair_names,
            bucket_size=bucket_size,
            generation=self._next_batch_generation(batch_id),
        )
        self.batches[batch_id] = batch_state

        # The gathered membership signature above validates that every pair
        # spans the same two sides before any batch state or process group is created.
        first_pair = self.pairs[batch_state.pair_names[0]]

        batch_state.local_leader = sorted(first_pair.local_ranks)[0]
        batch_state.local_group = first_pair.local_group
        batch_state.batch_group = first_pair.pair_group

        # Both sides materialize local send before recv, which are opposite
        # directions. Create their union before either can first-touch a group.
        prewarm_broadcast_groups(
            m2m
            for pair_name in batch_state.pair_names
            for m2m in (self.pairs[pair_name].m2m_send, self.pairs[pair_name].m2m_recv)
        )

        all_send_chunks = []
        all_recv_chunks = []

        # Process each pair in the validated canonical order.
        for pair_name in batch_state.pair_names:
            tensor_payloads = grouped[pair_name]
            pair_state = self.pairs[pair_name]

            logger.info(
                f"Agent {self.rank}: Batch {batch_id}: Registering {len(tensor_payloads)} tensors for pair '{pair_name}'"
            )

            # Initialize per-pair lists in BatchState
            batch_state.pair_tensors[pair_name] = []
            batch_state.pair_target_dtypes[pair_name] = []

            # Per-pair chunk lists (for bucketization)
            pair_send_chunks: list[Chunk] = []
            pair_recv_chunks: list[Chunk] = []

            for i, tensor_payload in enumerate(tensor_payloads):
                tensor = ForkingPickler.loads(tensor_payload)
                batch_state.pair_tensors[pair_name].append(tensor)

                # Exchange dtype information between rank0s
                my_rank0 = pair_state.local_ranks[0]
                target_rank0 = pair_state.remote_ranks[0]
                my_dtype_list = [tensor.dtype]
                target_dtype_list = [None]

                if self.rank == my_rank0:
                    if pair_state.local_is_first:
                        dist.send_object_list(my_dtype_list, dst=target_rank0)
                        dist.recv_object_list(target_dtype_list, src=target_rank0)
                    else:
                        dist.recv_object_list(target_dtype_list, src=target_rank0)
                        dist.send_object_list(my_dtype_list, dst=target_rank0)

                dist.broadcast_object_list(target_dtype_list, src=my_rank0, group=pair_state.local_group)
                target_dtype = target_dtype_list[0]
                batch_state.pair_target_dtypes[pair_name].append(target_dtype)
                logger.debug(f"Agent {self.rank}: Batch {batch_id}: tensor {i} target dtype {target_dtype}")

                if pair_state.m2m_send or pair_state.m2m_recv:
                    logger.debug(
                        f"Agent {self.rank}: Batch {batch_id}: Generating chunks for tensor {i} with shape {tensor.shape}"
                    )

                    # Calculate smart transfer_dtype: min(my_dtype, remote_dtype) by itemsize
                    transfer_dtype = None
                    if target_dtype is not None:
                        my_itemsize = tensor.dtype.itemsize
                        remote_itemsize = target_dtype.itemsize
                        transfer_dtype = tensor.dtype if my_itemsize <= remote_itemsize else target_dtype
                        logger.debug(
                            f"Agent {self.rank}: Batch {batch_id}: tensor {i} transfer_dtype={transfer_dtype} "
                            f"(my={tensor.dtype}, remote={target_dtype})"
                        )

                    if pair_state.m2m_send is not None:
                        send_chunks = m2m_to_chunks(
                            pair_state.m2m_send,
                            rank=self.rank,
                            source_tensor=tensor,
                            target_tensor=None,
                            transfer_dtype=transfer_dtype,
                            source_partial_groups=pair_state.source_partial_groups,
                        )
                        pair_send_chunks.extend(send_chunks)

                    if pair_state.m2m_recv is not None:
                        recv_chunks = m2m_to_chunks(
                            pair_state.m2m_recv,
                            rank=self.rank,
                            source_tensor=None,
                            target_tensor=tensor,
                            transfer_dtype=transfer_dtype,
                        )
                        pair_recv_chunks.extend(recv_chunks)

            # Accumulate to flattened lists
            all_send_chunks.extend(pair_send_chunks)
            all_recv_chunks.extend(pair_recv_chunks)

            logger.info(f"Agent {self.rank}: Batch {batch_id}: Completed registration for pair '{pair_name}'")

        # Bucketize (cross-pair, by channel key) into BatchState. bucket_size
        # unset means no coalescing: every chunk becomes its own single-entry
        # bucket. Chunks are only an intermediate; only buckets are executed.
        coalesce_bytes = bucket_size or 1
        batch_state.send_buckets = chunk_to_bucket_ops(chunks=all_send_chunks, bucket_size=coalesce_bytes)
        batch_state.recv_buckets = chunk_to_bucket_ops(chunks=all_recv_chunks, bucket_size=coalesce_bytes)
        logger.info(
            f"Agent {self.rank}: Batch {batch_id}: Unified buckets: "
            f"send ({len(batch_state.send_buckets)} buckets), recv ({len(batch_state.recv_buckets)} buckets)"
        )

        logger.info(
            f"Agent {self.rank}: Batch {batch_id}: Registration complete - "
            f"{len(tensors)} tensors across {len(grouped)} pairs"
        )
        if msg.semaphore_name:
            self._write_command_result(msg.semaphore_name, batch_state.generation)
        return True

    def _register_dual_role(
        self, batch_state: BatchState, grouped: dict[str, list[memoryview]], role: str, semaphore_name: str | None
    ) -> bool:
        """Merge one role's registration into a dual-endpoint batch.

        Chunks are generated once both roles are present for every pair of
        the batch (see _generate_dual_buckets). Returns True when generation
        ran; False when the batch still waits for the other role, in which
        case ``semaphore_name`` is parked on the batch for release at
        completion (the caller must not release it).
        """
        batch_id = batch_state.batch_id

        first_pair = self.pairs[batch_state.pair_names[0]]
        first_name = min(first_pair.role_ranks)  # canonical, same on every agent
        batch_state.local_leader = sorted(first_pair.role_ranks[first_name])[0]
        batch_state.local_group = first_pair.role_groups[first_name]
        batch_state.batch_group = first_pair.pair_group

        for pair_name in batch_state.pair_names:
            if (pair_name, role) in batch_state.pair_role_tensors:
                raise _InvalidRegistrationError(
                    f"Batch {batch_id}: role '{role}' already registered for pair '{pair_name}'; "
                    f"dual-endpoint batches merge roles and never overwrite"
                )
            tensors_role = [ForkingPickler.loads(payload) for payload in grouped[pair_name]]
            batch_state.pair_role_tensors[(pair_name, role)] = tensors_role
            batch_state.pair_role_dtypes[(pair_name, role)] = [t.dtype for t in tensors_role]
            logger.info(
                f"Agent {self.rank}: Batch {batch_id}: Registered {len(tensors_role)} tensors for "
                f"pair '{pair_name}' role '{role}'"
            )

        # Both roles present for every pair?
        for pair_name in batch_state.pair_names:
            roles_present = {r for (p, r) in batch_state.pair_role_tensors if p == pair_name}
            if len(roles_present) < 2:
                logger.info(
                    f"Agent {self.rank}: Batch {batch_id}: waiting for the other role of pair "
                    f"'{pair_name}' (have {roles_present})"
                )
                if semaphore_name:
                    batch_state.pending_register_semaphores.append(semaphore_name)
                return False

        try:
            self._generate_dual_buckets(batch_state)
        except Exception as e:
            self._abort_dual_registration(batch_state, e)
            raise
        for parked in batch_state.pending_register_semaphores:
            self._write_command_result(parked, batch_state.generation)
            self._release_semaphore(parked)
        batch_state.pending_register_semaphores.clear()
        if semaphore_name:
            self._write_command_result(semaphore_name, batch_state.generation)
        return True

    def _next_batch_generation(self, batch_id: str) -> int:
        gens = getattr(self, "_batch_generation", None)
        if gens is None:
            gens = self._batch_generation = {}
        gen = gens.get(batch_id, 0) + 1
        gens[batch_id] = gen
        if getattr(self, "state_env", None) is not None:
            with self.state_env.begin(write=True, db=self.state_db) as txn:
                txn.put(f"batch:{batch_id}/generation".encode(), msgspec.msgpack.encode(gen))
        return gen

    def _abort_dual_registration(self, batch: BatchState, error: BaseException) -> None:
        """Fail a dual batch: wake parked registers and drop the incomplete state."""
        for parked in batch.pending_register_semaphores:
            self._record_command_error(parked, error)
            self._release_semaphore(parked)
        batch.pending_register_semaphores.clear()
        self.batches.pop(batch.batch_id, None)

    def _generate_dual_buckets(self, batch_state: BatchState):
        """Generate per-direction buckets for a fully-registered dual batch.

        For each direction (src_role -> dst_role) every tensor index produces
        source chunks and target chunks from the same map in one call, so the
        same-rank overlap surfaces as LOCAL chunks. Order is canonical (pairs
        sorted, direction by source-role name, tensors in registration order)
        and identical on every rank.
        """
        batch_id = batch_state.batch_id
        coalesce_bytes = batch_state.bucket_size or 1

        # Both directions may broadcast; create the union of groups before any
        # direction can first-touch one (same contract as the split path).
        prewarm_broadcast_groups(
            m2m
            for pair_name in batch_state.pair_names
            for m2m in (self.pairs[pair_name].m2m_by_source_role or {}).values()
        )

        # Chunks accumulate across ALL pairs of the batch per direction before
        # a single bucketization: assigning per pair would keep only the LAST
        # pair's buckets for a direction and silently drop every other pair's
        # weights. Chunk order is canonical (pairs sorted, directions by
        # source-role name, tensors in registration order) and identical on
        # every rank, so bucket boundaries are deterministic too.
        chunks_by_direction: dict[tuple[str, str], list[Chunk]] = {}
        for pair_name in sorted(batch_state.pair_names):
            pair = self.pairs[pair_name]
            for src_role in sorted(pair.m2m_by_source_role):
                dst_roles = [name for name in pair.m2m_by_source_role if name != src_role]
                if not dst_roles:
                    raise ValueError(f"Pair '{pair_name}' has a single peer name; not dual")
                dst_role = dst_roles[0]
                m2m = pair.m2m_by_source_role[src_role]
                if m2m is None or (
                    (src_role, dst_role) in chunks_by_direction and chunks_by_direction[(src_role, dst_role)] is None
                ):
                    logger.info(
                        f"Agent {self.rank}: Batch {batch_id}: pair '{pair_name}' has no map for direction "
                        f"{src_role}->{dst_role} (Partial target unsupported)"
                    )
                    chunks_by_direction[(src_role, dst_role)] = None
                    continue
                partials = (pair.partial_by_source_role or {}).get(src_role)
                src_tensors = batch_state.pair_role_tensors[(pair_name, src_role)]
                dst_tensors = batch_state.pair_role_tensors[(pair_name, dst_role)]
                src_dtypes = batch_state.pair_role_dtypes[(pair_name, src_role)]
                dst_dtypes = batch_state.pair_role_dtypes[(pair_name, dst_role)]
                if len(src_tensors) != len(dst_tensors):
                    raise _InvalidRegistrationError(
                        f"Batch {batch_id}: pair '{pair_name}' role tensor-count mismatch: "
                        f"{src_role}={len(src_tensors)}, {dst_role}={len(dst_tensors)}"
                    )

                chunks = chunks_by_direction.setdefault((src_role, dst_role), [])
                for i in range(len(src_tensors)):
                    # Wire dtype: min(itemsize) of the two roles, source side casts
                    if src_dtypes[i].itemsize <= dst_dtypes[i].itemsize:
                        transfer_dtype = src_dtypes[i]
                    else:
                        transfer_dtype = dst_dtypes[i]
                    chunks.extend(
                        m2m_to_chunks(
                            m2m,
                            rank=self.rank,
                            source_tensor=src_tensors[i],
                            target_tensor=dst_tensors[i],
                            transfer_dtype=transfer_dtype,
                            source_partial_groups=partials,
                        )
                    )
                logger.info(
                    f"Agent {self.rank}: Batch {batch_id}: pair '{pair_name}' direction "
                    f"{src_role}->{dst_role}: {len(chunks)} chunks so far"
                )

        for direction in sorted(chunks_by_direction):
            chunks = chunks_by_direction[direction]
            if chunks is None:
                batch_state.dual_direction_buckets[direction] = None
                continue
            batch_state.dual_direction_buckets[direction] = chunk_to_bucket_ops(
                chunks=chunks, bucket_size=coalesce_bytes
            )
            logger.info(
                f"Agent {self.rank}: Batch {batch_id}: direction {direction[0]}->{direction[1]}: "
                f"{len(batch_state.dual_direction_buckets[direction])} buckets "
                f"({len(chunks)} chunks across {len(batch_state.pair_names)} pairs)"
            )

        logger.info(
            f"Agent {self.rank}: Batch {batch_id}: Dual registration complete - "
            f"directions: {sorted(batch_state.dual_direction_buckets)}"
        )

    def _collect_mesh_placement_info(
        self, pair_name: str, ranks: list[int], name: str
    ) -> list[tuple[tuple[int, ...], tuple[Placement, ...]]]:
        """Collect one side's mesh shape and placement info from its ranks.

        ``name`` selects the side (peer name): mesh metadata keys carry the
        peer name so a dual-endpoint agent can hold both sides' meshes.
        """
        mesh_info_list = []

        for rank in ranks:
            mesh_shape_key = f"pair:{pair_name}/rank:{rank}/{name}/mesh_shape"
            placements_key = f"pair:{pair_name}/rank:{rank}/{name}/placements"

            mesh_shape_bytes = self.store.get_bytes(mesh_shape_key)
            placements_bytes = self.store.get_bytes(placements_key)

            if mesh_shape_bytes is not None and placements_bytes is not None:
                mesh_shape = ForkingPickler.loads(mesh_shape_bytes)
                placements = ForkingPickler.loads(placements_bytes)
                mesh_info_list.append((mesh_shape, placements))

        return mesh_info_list

    def _validate_mesh_placement_consistency(self, mesh_info_list: list[tuple[tuple[int, ...], tuple[Placement, ...]]]):
        """Validate that all ranks have consistent mesh/placement configuration."""
        if len(mesh_info_list) == 1:
            return

        for i, mesh_info in enumerate(mesh_info_list):
            # Check mesh shape consistency
            assert mesh_info == mesh_info_list[0], (
                f"Agent {self.rank}: rank {i} mesh info {mesh_info} != reference {mesh_info_list[0]}"
            )

    def _extract_rank(self, key: str) -> int:
        """Extract rank number from key like 'pair:foo/rank:3/bar'."""
        idx = key.find("/rank:")
        if idx == -1:
            raise ValueError(f"No rank found in key: {key}")
        rest = key[idx + 6 :]  # skip "/rank:"
        return int(rest.split("/")[0])

    def _update_heartbeat(self):
        """Update heartbeat timestamp in State LMDB.

        This allows Workers to verify the Agent is alive and responsive.
        Called on startup and every main loop iteration.
        """
        with self.state_env.begin(write=True, db=self.state_db) as txn:
            txn.put(b"agent:heartbeat", str(time.time()).encode())

    def _leader_set(self, key: str, value: str, batch: BatchState, component: str = "global") -> None:
        """Set key-value where only leader writes.

        All ranks in local group synchronize after write.
        """
        if self.rank == batch.local_leader:
            self.store.set(key, value, component=component)
        dist.barrier(batch.local_group)

    def _write_command_result(self, semaphore_name: str, value: int | bool) -> None:
        now = time.monotonic()
        expired = [
            name for name, written_at in self._command_error_times.items() if now - written_at > COMMAND_ERROR_TTL
        ]
        with self.state_env.begin(write=True, db=self.state_db) as txn:
            for name in expired:
                txn.delete(command_error_key(name))
                txn.delete(command_result_key(name))
            txn.put(command_result_key(semaphore_name), msgspec.msgpack.encode(value))
        for name in expired:
            del self._command_error_times[name]
        self._command_error_times[semaphore_name] = now

    def _record_command_error(self, semaphore_name: str, error: Exception) -> None:
        payload = f"{type(error).__name__}: {error}"
        now = time.monotonic()
        expired = [
            name for name, written_at in self._command_error_times.items() if now - written_at > COMMAND_ERROR_TTL
        ]
        with self.state_env.begin(write=True, db=self.state_db) as txn:
            for name in expired:
                txn.delete(command_error_key(name))
            txn.put(command_error_key(semaphore_name), msgspec.msgpack.encode(payload))
        for name in expired:
            del self._command_error_times[name]
        self._command_error_times[semaphore_name] = now

    def _release_semaphore(self, semaphore_name: str):
        try:
            # Open the semaphore (must be created by client)
            sem = posix_ipc.Semaphore(semaphore_name)
        except posix_ipc.ExistentialError:
            logger.warning(f"Agent {self.rank}: Semaphore '{semaphore_name}' not found")
            return
        except Exception as e:
            logger.error(f"Agent {self.rank}: Error opening semaphore '{semaphore_name}': {e}")
            return

        try:
            sem.release()
            logger.debug(f"Agent {self.rank}: Released semaphore '{semaphore_name}'")
        except Exception as e:
            logger.error(f"Agent {self.rank}: Error releasing semaphore '{semaphore_name}': {e}")

        try:
            sem.close()
            sem.unlink()
            logger.debug(f"Agent {self.rank}: Closed and unlinked semaphore '{semaphore_name}'")
        except posix_ipc.ExistentialError:
            logger.debug(f"Agent {self.rank}: Semaphore '{semaphore_name}' already unlinked")
        except Exception as e:
            logger.error(f"Agent {self.rank}: Error closing semaphore '{semaphore_name}': {e}")

    def close(self, destroy: bool = True):
        """Cleanup resources."""
        self.command_queue.close(destroy=destroy)
        if self.state_env:
            self.state_env.close()
        if destroy:
            try:
                self.lmdb_state_path.unlink()
            except Exception:
                pass
        self.store.close()
        dist.destroy_process_group()
