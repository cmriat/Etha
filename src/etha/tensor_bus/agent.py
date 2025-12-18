"""Tensor Bus Agent Process."""

import os
import time
import uuid
import logging
import traceback
from multiprocessing.reduction import ForkingPickler

import lmdb
import torch
import msgspec
import posix_ipc
import torch.distributed as dist
from upath import UPath
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

from etha.comm import (
    chunk_comm,
    bucket_comm,
    get_m2m_map,
    map_to_chunk_ops,
    chunk_to_bucket_ops,
)
from etha.comm.ir import Chunk
from etha.kvstore import KVStore, create_store
from etha.pg_utils import get_or_create_process_group

from .utils import setup_cuda_rebuild_patch
from .commands import InitPair, Transfer, QueryStatus, CleanupBatch, RegisterTensors
from .pair_state import M2MMap, PairState
from .batch_state import BatchState
from .command_queue import CommandQueue

logger = logging.getLogger(__name__)

TIME_INTERVAL = 0.001  # 1ms


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
        """
        self.rank = rank
        self.world_size = world_size

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        # Initialize torch.distributed first (needed for namespace broadcast)
        logger.debug(f"Agent {rank}: Initializing torch.distributed")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

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
        logger.info(f"Agent {rank}: State LMDB initialized at {lmdb_state_path}")

        # Write initial heartbeat (for connection validation)
        self._update_heartbeat()
        logger.debug(f"Agent {rank}: Initial heartbeat written")

        self.pairs: dict[str, PairState] = {}
        self.batches: dict[str, BatchState] = {}

        setup_cuda_rebuild_patch()

        logger.info(f"Agent {rank}: Initialized successfully")

    def run(self):
        """Main loop: process commands from Host."""
        logger.info(f"Agent {self.rank}: Starting main loop")

        while True:
            # Update heartbeat (for connection validation)
            self._update_heartbeat()

            msg = self.command_queue.dequeue(block=True, timeout=TIME_INTERVAL)
            if msg is not None:
                self._handle_command(msg)

    def _handle_command(self, command):
        """Dispatch command to appropriate handler and handle semaphore release."""
        try:
            match command:
                case InitPair():
                    self._handle_init_pair(command)
                case Transfer():
                    self._handle_transfer(command)
                case QueryStatus():
                    self._handle_query_status(command)
                case RegisterTensors():
                    self._handle_register_tensors(command)
                case CleanupBatch():
                    self._handle_cleanup_batch(command)
                case _:
                    logger.warning(f"Agent {self.rank}: Unknown command type: {type(command)}")
                    return  # Don't release semaphore for unknown commands

            # Release semaphore if specified
            if command.semaphore_name:
                self._release_semaphore(command.semaphore_name)

        except Exception as e:
            logger.error(f"Agent {self.rank}: Error handling command {type(command)}: {e} {traceback.format_exc()}")
            # Still try to release semaphore even on error to avoid client hanging
            if command.semaphore_name:
                self._release_semaphore(command.semaphore_name)
            raise

    def _handle_init_pair(self, msg: InitPair):
        """Handle RegisterPair command.

        Steps:
        1. Write to KVStore ready local registration
        2. Write expected_world_size (if first)
        3. Write device mesh and placement info to KVStore
        4. Poll KVStore until my side is complete
        5. Poll KVStore until remote side is complete
        6. Exchange and collect device mesh/placement info from all ranks
        7. Create PairState with m2m maps info
        """
        pair_name = msg.pair_name
        local_name = msg.local_name
        expected_local = msg.expected_world_size
        remote_name = msg.remote_name

        logger.info(f"Agent {self.rank}: RegisterPair pair={pair_name}, local={local_name} -> remote={remote_name}")

        # Step 1: Write local registration to KVStore
        logger.info(f"Agent {self.rank}: About to write local registration to KVStore")
        local_key = f"pair:{pair_name}/rank:{self.rank}/{local_name}"
        self.store.set(local_key, "1")
        logger.info(f"Agent {self.rank}: Wrote {local_key} = '1' to KVStore")

        # Step 2: Write expected_world_size (all ranks write the same value, idempotent)
        logger.info(f"Agent {self.rank}: About to write expected_world_size")
        expected_key = f"pair:{pair_name}/{local_name}/expected_world_size"
        self.store.set(expected_key, str(expected_local))
        logger.info(f"Agent {self.rank}: Wrote {expected_key}={expected_local}")

        # Step 3: Write device mesh and placement info to store
        if msg.mesh_shape_payload is not None and msg.placements_payload is not None:
            logger.info(f"Agent {self.rank}: About to write device mesh info")
            mesh_shape_key = f"pair:{pair_name}/rank:{self.rank}/mesh_shape"
            self.store.set_bytes(mesh_shape_key, bytes(msg.mesh_shape_payload))
            logger.info(f"Agent {self.rank}: Wrote mesh_shape to store")

            placements_key = f"pair:{pair_name}/rank:{self.rank}/placements"
            self.store.set_bytes(placements_key, bytes(msg.placements_payload))
            logger.info(f"Agent {self.rank}: Wrote placements to store")
        else:
            logger.info(f"Agent {self.rank}: No mesh/placement info to write")

        # Step 4: Wait until local peer is complete
        logger.info(f"Agent {self.rank}: Waiting for local peer '{local_name}' (expected={expected_local})")
        local_keys = self.store.wait_for_keys(
            key_pattern=f"pair:{pair_name}/rank:*/{local_name}",
            expected_count=expected_local,
            candidate_keys=[f"pair:{pair_name}/rank:{r}/{local_name}" for r in range(self.world_size)],
        )
        local_ranks = sorted([self._extract_rank(k) for k in local_keys])
        logger.info(f"Agent {self.rank}: Local peer complete: {local_ranks}")

        # Step 5: Wait until remote peer has written expected_world_size
        logger.info(f"Agent {self.rank}: Waiting for remote peer '{remote_name}'")

        remote_expected_key = f"pair:{pair_name}/{remote_name}/expected_world_size"
        expected_remote = int(self.store.wait_for_key(remote_expected_key).decode())
        logger.debug(f"Agent {self.rank}: Remote peer '{remote_name}' expects {expected_remote} ranks")

        # Then, wait for all remote ranks to register
        remote_keys = self.store.wait_for_keys(
            key_pattern=f"pair:{pair_name}/rank:*/{remote_name}",
            expected_count=expected_remote,
            candidate_keys=[f"pair:{pair_name}/rank:{r}/{remote_name}" for r in range(self.world_size)],
        )
        remote_ranks = sorted([self._extract_rank(k) for k in remote_keys])

        logger.info(f"Agent {self.rank}: Remote peer '{remote_name}' complete: {remote_ranks}")
        logger.debug(f"Agent {self.rank}: Creating pair group with ranks: {local_ranks + remote_ranks}")
        pair_group = get_or_create_process_group(local_ranks + remote_ranks)
        logger.debug(f"Agent {self.rank}: Pair group created: {pair_group}")

        # Step 6: Collect device mesh and placement info from all ranks
        local_mesh_info = self._collect_mesh_placement_info(pair_name, local_ranks)
        remote_mesh_info = self._collect_mesh_placement_info(pair_name, remote_ranks)

        # Step 7: Validate mesh/placement consistency
        if local_mesh_info:
            self._validate_mesh_placement_consistency(local_mesh_info)

        if remote_mesh_info:
            self._validate_mesh_placement_consistency(remote_mesh_info)

        # Step 8: Generate P2P maps if validation passed
        m2m_map_send = None
        m2m_map_recv = None

        # Determine canonical ordering to ensure all ranks call get_m2m_map in same order
        # This prevents deadlock in collective operations within get_m2m_map
        # We use alphabetical order of role names as the tie-breaker
        local_is_first = local_name < remote_name

        if local_mesh_info and remote_mesh_info:
            # Get local mesh info (this process's mesh)
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

            if local_is_first:
                first_mesh = DeviceMesh("cuda", local_mesh_tensor)
                second_mesh = DeviceMesh("cuda", remote_mesh_tensor)
                first_placements = local_placements
                second_placements = remote_placements
            else:
                first_mesh = DeviceMesh("cuda", remote_mesh_tensor)
                second_mesh = DeviceMesh("cuda", local_mesh_tensor)
                first_placements = remote_placements
                second_placements = local_placements
            logger.info(f"Agent {self.rank}: Generating M2M maps for pair '{pair_name}'")

            # Generate M2M Maps (topology layer, shape-independent)
            # IMPORTANT: All ranks must call in the same order to avoid deadlock
            # First call: first_mesh -> second_mesh
            map_1, src_slicers_1, tgt_slicers_1 = get_m2m_map(
                source_mesh=first_mesh,
                source_placements=first_placements,
                target_mesh=second_mesh,
                target_placements=second_placements,
                group=pair_group,
                device="cuda",
            )
            # Second call: second_mesh -> first_mesh
            map_2, src_slicers_2, tgt_slicers_2 = get_m2m_map(
                source_mesh=second_mesh,
                source_placements=second_placements,
                target_mesh=first_mesh,
                target_placements=first_placements,
                group=pair_group,
                device="cuda",
            )

            # Assign to send/recv based on which mesh is local
            if local_is_first:
                m2m_map_send = M2MMap(
                    m2m_map=map_1,
                    source_num_slicers=src_slicers_1,
                    target_num_slicers=tgt_slicers_1,
                )
                m2m_map_recv = M2MMap(
                    m2m_map=map_2,
                    source_num_slicers=src_slicers_2,
                    target_num_slicers=tgt_slicers_2,
                )
            else:
                m2m_map_send = M2MMap(
                    m2m_map=map_2,
                    source_num_slicers=src_slicers_2,
                    target_num_slicers=tgt_slicers_2,
                )
                m2m_map_recv = M2MMap(
                    m2m_map=map_1,
                    source_num_slicers=src_slicers_1,
                    target_num_slicers=tgt_slicers_1,
                )
            logger.info(
                f"Agent {self.rank}: Generated P2P maps for pair '{pair_name}'. m2m_map_send: {m2m_map_send} m2m_map_recv: {m2m_map_recv}"
            )
        else:
            logger.info(f"Agent {self.rank}: Skipping P2P map generation - missing or inconsistent mesh/placement info")

        # Step 9: Create PairState
        if local_is_first:
            local_group = get_or_create_process_group(local_ranks)
            get_or_create_process_group(remote_ranks)
        else:
            get_or_create_process_group(remote_ranks)
            local_group = get_or_create_process_group(local_ranks)

        state = PairState(
            pair_name=pair_name,
            local_name=local_name,
            local_ranks=local_ranks,
            remote_name=remote_name,
            remote_ranks=remote_ranks,
            pair_size=expected_local + expected_remote,
            local_group=local_group,
            pair_group=pair_group,
            local_is_first=local_is_first,
            m2m_send=m2m_map_send,
            m2m_recv=m2m_map_recv,
        )
        self.pairs[pair_name] = state

        # Step 10: Write PairState to State LMDB (for Worker verification)
        state_key = f"pair:{pair_name}/state:match".encode()  # LMDB key
        state_bytes = msgspec.msgpack.encode("matched")
        with self.state_env.begin(write=True, db=self.state_db) as txn:
            txn.put(state_key, state_bytes)
        logger.debug(f"Agent {self.rank}: Wrote PairState to State LMDB")

        logger.info(
            f"Agent {self.rank}: Pair '{pair_name}' matched! "
            f"Local '{local_name}': {local_ranks}, Remote '{remote_name}': {remote_ranks}"
        )

    def _handle_transfer(self, msg: Transfer):
        """Handle Transfer command for batch tensor transfer."""
        batch_id = msg.batch_id
        transfer_type = msg.transfer_type
        logger.info(f"Agent {self.rank}: Handling transfer for batch '{batch_id}' ({transfer_type})")

        if batch_id not in self.batches:
            raise ValueError(f"Transfer for unknown batch: {batch_id}")

        batch_state = self.batches[batch_id]

        # Set transfer_signal to notify receiver that sender is ready (before barrier)
        transfer_signal_key = f"batch:{batch_id}/state:transfer_signal"
        self._leader_set(transfer_signal_key, "1", batch_state)
        logger.info(
            f"Agent {self.rank}: set key {self.store._prefixed(transfer_signal_key, component='global')} value 1"
        )
        # Synchronize all ranks in batch
        dist.barrier(batch_state.batch_group)
        logger.debug(f"Agent {self.rank}: Batch {batch_id}: All ranks synchronized")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Execute transfer using flattened chunks/buckets
        if batch_state.send_chunks or batch_state.recv_chunks:
            if transfer_type == "send":
                buckets = batch_state.send_buckets
                chunks = batch_state.send_chunks
            else:
                buckets = batch_state.recv_buckets
                chunks = batch_state.recv_chunks

            if buckets:
                logger.info(
                    f"Agent {self.rank}: Batch {batch_id}: Executing bucketized transfer with {len(buckets)} buckets"
                )
                bucket_comm(buckets=buckets)
            elif chunks:
                logger.info(
                    f"Agent {self.rank}: Batch {batch_id}: Executing chunk-based transfer with {len(chunks)} chunks"
                )
                chunk_comm(chunks=chunks)
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

        end_event.record()
        torch.cuda.synchronize()
        transfer_time_ms = start_event.elapsed_time(end_event)

        dist.barrier(batch_state.batch_group)
        self._leader_set(transfer_signal_key, "0", batch_state)
        logger.info(f"Agent {self.rank}: Batch {batch_id}: Transfer complete in {transfer_time_ms:.2f} ms")

    def _handle_query_status(self, msg: QueryStatus):
        batch_id = msg.batch_id
        state_name = msg.state_name  # e.g. "transfer_signal"
        store_state_key = f"batch:{batch_id}/state:{state_name}"
        statedb_key = f"batch:{batch_id}/state:{state_name}".encode()  # LMDB key

        if batch_id not in self.batches:
            logger.error(f"Agent {self.rank}: QueryStatus for unknown batch: {batch_id}")
            return

        batch_state = self.batches[batch_id]

        if state_name == "transfer_signal":
            # Leader reads from store, broadcasts to others
            raw_value = self._leader_get(store_state_key, batch_state)
            state = raw_value == b"1"
            logger.info(f"Agent {self.rank}: Query {state_name} batch={batch_id}: raw={raw_value}, state={state}")
        else:
            logger.error(f"Agent {self.rank}: Invalid state name: {state_name}")
            return

        with self.state_env.begin(write=True, db=self.state_db) as txn:
            txn.put(statedb_key, msgspec.msgpack.encode(state))

    def _handle_cleanup_batch(self, msg: CleanupBatch):
        """Handle CleanupBatch command to free batch state resources."""
        batch_id = msg.batch_id

        if batch_id in self.batches:
            del self.batches[batch_id]
            logger.info(f"Agent {self.rank}: Cleaned up batch {batch_id}")
        else:
            logger.warning(f"Agent {self.rank}: Cleanup requested for unknown batch {batch_id}")

    def _handle_register_tensors(self, msg: RegisterTensors):
        """Handle RegisterTensors command for batch tensor registration.

        Creates a new BatchState with flattened chunks/buckets across all pairs.
        """
        batch_id = msg.batch_id
        tensors = msg.tensors  # list[tuple[str, memoryview]]
        bucket_size = msg.bucket_size

        logger.info(f"Agent {self.rank}: Starting batch {batch_id}: {len(tensors)} tensors")

        # Group tensors by pair_name
        grouped: dict[str, list[memoryview]] = {}
        for pair_name, tensor_payload in tensors:
            if pair_name not in grouped:
                grouped[pair_name] = []
            grouped[pair_name].append(tensor_payload)

        batch_state = BatchState(
            batch_id=batch_id,
            pair_names=list(grouped.keys()),
            bucket_size=bucket_size,
        )
        self.batches[batch_id] = batch_state

        # Validate all pairs have same local_ranks and remote_ranks, set batch-level groups
        first_pair = self.pairs[batch_state.pair_names[0]]
        first_local_ranks = set(first_pair.local_ranks)
        first_remote_ranks = set(first_pair.remote_ranks)

        for pair_name in batch_state.pair_names[1:]:
            pair = self.pairs[pair_name]
            if set(pair.local_ranks) != first_local_ranks or set(pair.remote_ranks) != first_remote_ranks:
                raise ValueError(
                    f"Batch {batch_id}: pair '{pair_name}' has different ranks"
                    f"({pair.local_ranks}, {pair.remote_ranks}) than first pair ({first_pair.local_ranks}, {first_pair.remote_ranks})"
                )

        batch_state.local_leader = sorted(first_pair.local_ranks)[0]
        batch_state.local_group = first_pair.local_group
        batch_state.batch_group = first_pair.pair_group

        all_send_chunks = []
        all_recv_chunks = []

        # Process each pair
        for pair_name, tensor_payloads in grouped.items():
            if pair_name not in self.pairs:
                raise ValueError(f"RegisterTensors for unknown pair: {pair_name}")

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

                if pair_state.m2m_send and pair_state.m2m_recv:
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

                    # Generate send chunks for this tensor
                    send_chunks = map_to_chunk_ops(
                        m2m_map=pair_state.m2m_send.m2m_map,
                        rank=self.rank,
                        source_num_slicers=pair_state.m2m_send.source_num_slicers,
                        target_num_slicers=pair_state.m2m_send.target_num_slicers,
                        source_tensor=tensor,
                        target_tensor=None,
                        transfer_dtype=transfer_dtype,
                    )
                    pair_send_chunks.extend(send_chunks)

                    # Generate recv chunks for this tensor
                    recv_chunks = map_to_chunk_ops(
                        m2m_map=pair_state.m2m_recv.m2m_map,
                        rank=self.rank,
                        source_num_slicers=pair_state.m2m_recv.source_num_slicers,
                        target_num_slicers=pair_state.m2m_recv.target_num_slicers,
                        source_tensor=None,
                        target_tensor=tensor,
                        transfer_dtype=transfer_dtype,
                    )
                    pair_recv_chunks.extend(recv_chunks)

            # Accumulate to flattened lists
            all_send_chunks.extend(pair_send_chunks)
            all_recv_chunks.extend(pair_recv_chunks)

            logger.info(f"Agent {self.rank}: Batch {batch_id}: Completed registration for pair '{pair_name}'")

        # Store flattened chunks and buckets in BatchState
        batch_state.send_chunks = all_send_chunks
        batch_state.recv_chunks = all_recv_chunks

        logger.info(
            f"Agent {self.rank}: Batch {batch_id}: Flattened chunks: "
            f"send ({len(all_send_chunks)}), recv ({len(all_recv_chunks)})"
        )

        # Unified bucketization (cross-pair, by channel key)
        if bucket_size:
            batch_state.send_buckets = chunk_to_bucket_ops(
                chunks=all_send_chunks,
                bucket_size=bucket_size,
            )
            batch_state.recv_buckets = chunk_to_bucket_ops(
                chunks=all_recv_chunks,
                bucket_size=bucket_size,
            )
            logger.info(
                f"Agent {self.rank}: Batch {batch_id}: Unified buckets: "
                f"send ({len(batch_state.send_buckets)} buckets), recv ({len(batch_state.recv_buckets)} buckets)"
            )

        logger.info(
            f"Agent {self.rank}: Batch {batch_id}: Registration complete - "
            f"{len(tensors)} tensors across {len(grouped)} pairs"
        )

    def _collect_mesh_placement_info(
        self, pair_name: str, ranks: list[int]
    ) -> list[tuple[tuple[int, ...], tuple[Placement, ...]]]:
        """Collect mesh shape and placement info from ranks."""
        mesh_info_list = []

        for rank in ranks:
            mesh_shape_key = f"pair:{pair_name}/rank:{rank}/mesh_shape"
            placements_key = f"pair:{pair_name}/rank:{rank}/placements"

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

    def _leader_get(self, key: str, batch: BatchState, component: str = "global") -> bytes | None:
        """Get key-value where only leader reads, then broadcasts.

        Returns:
            Value bytes from store (same for all ranks in local group)
        """
        if self.rank == batch.local_leader:
            value = self.store.get(key, component=component)
        else:
            value = None

        result = [value]
        dist.broadcast_object_list(result, src=batch.local_leader, group=batch.local_group)
        return result[0]

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
