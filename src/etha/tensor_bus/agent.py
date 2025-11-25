"""Tensor Bus Agent Process."""

import os
import time
import base64
import logging
import traceback
from datetime import timedelta
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
        tcpstore_host: str,
        tcpstore_port: int,
        lmdb_command_queue_path: str,
        lmdb_state_path: str,
    ):
        """Initialize Agent.

        Args:
            rank: Rank in the torch.distributed group
            world_size: Total number of Agents
            tcpstore_host: TCPStore server address
            tcpstore_port: TCPStore server port
            lmdb_command_queue_path: Path to CommandQueue LMDB
            lmdb_state_path: Path to State LMDB
        """
        self.rank = rank
        self.world_size = world_size
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        # Initialize torch.distributed
        logger.info(f"Agent {rank}: Initializing torch.distributed")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        # Initialize TCPStore
        logger.info(f"Agent {rank}: Connecting to TCPStore at {tcpstore_host}:{tcpstore_port}")
        self.store = dist.TCPStore(
            host_name=tcpstore_host,
            port=tcpstore_port,
            world_size=world_size,
            is_master=(rank == 0),  # Rank 0 is master
            timeout=timedelta(seconds=3600),
            wait_for_workers=True,
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
        logger.info(f"Agent {rank}: Initial heartbeat written")

        # Topology registry
        self.pairs: dict[str, PairState] = {}

        # Batch registry
        self.batches: dict[str, BatchState] = {}

        # Setup CUDA rebuild_cuda_tensor patch (once per process)
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
        1. Write to TCPStore: pair:{pair_name}:{side_name}:rank{rank} = "1"
        2. Write expected_world_size (if first)
        3. Write device mesh and placement info to TCPStore via base64 encoding
        4. Poll TCPStore until my side is complete
        5. Poll TCPStore until remote side is complete
        6. Exchange and collect device mesh/placement info from all ranks
        7. Create PairState with m2m maps info
        """
        pair_name = msg.pair_name
        local_name = msg.local_name
        expected_local = msg.expected_world_size
        remote_name = msg.remote_name

        logger.info(f"Agent {self.rank}: RegisterPair pair={pair_name}, local={local_name} -> remote={remote_name}")

        # Step 1: Write local registration to TCPStore
        logger.info(f"Agent {self.rank}: About to write local registration to TCPStore")
        local_key = f"pair:{pair_name}/rank:{self.rank}/{local_name}"
        self.store.set(local_key, "1")
        logger.info(f"Agent {self.rank}: Wrote {local_key} = '1' to TCPStore")

        # Step 2: Write expected_world_size (all ranks write the same value, idempotent)
        logger.info(f"Agent {self.rank}: About to write expected_world_size")
        expected_key = f"pair:{pair_name}/{local_name}/expected_world_size"
        self.store.set(expected_key, str(expected_local))
        logger.info(f"Agent {self.rank}: Wrote {expected_key}={expected_local}")

        # Step 3: Write device mesh and placement info to TCPStore
        if msg.mesh_shape_payload is not None and msg.placements_payload is not None:
            logger.info(f"Agent {self.rank}: About to write device mesh info")
            # Convert memoryview to bytes, then to base64 string for TCPStore
            mesh_shape_key = f"pair:{pair_name}/rank:{self.rank}/mesh_shape"
            mesh_shape_bytes = bytes(msg.mesh_shape_payload)
            mesh_shape_b64 = base64.b64encode(mesh_shape_bytes).decode("ascii")
            logger.info(f"Agent {self.rank}: mesh_shape_b64 length: {len(mesh_shape_b64)}, about to store.set")
            self.store.set(mesh_shape_key, mesh_shape_b64)
            logger.info(f"Agent {self.rank}: Wrote mesh_shape to TCPStore")

            placements_key = f"pair:{pair_name}/rank:{self.rank}/placements"
            placements_bytes = bytes(msg.placements_payload)
            placements_b64 = base64.b64encode(placements_bytes).decode("ascii")
            logger.info(f"Agent {self.rank}: placements_b64 length: {len(placements_b64)}, about to store.set")
            self.store.set(placements_key, placements_b64)
            logger.info(f"Agent {self.rank}: Wrote placements to TCPStore")
        else:
            logger.info(f"Agent {self.rank}: No mesh/placement info to write")

        # Step 4: Poll until local peer is complete
        logger.info(f"Agent {self.rank}: Waiting for local peer '{local_name}' to complete (expected={expected_local})")
        local_ranks = self._wait_until_ranks_ready(
            key_pattern=f"pair:{pair_name}/rank:{{rank}}/{local_name}",
            candidate_ranks=set(range(self.world_size)),
            expected_count=expected_local,
            description=f"Waiting for local peer '{local_name}'",
        )
        logger.info(f"Agent {self.rank}: Local peer complete: {local_ranks}")

        # Step 5: Poll until remote peer is complete
        logger.info(f"Agent {self.rank}: Waiting for remote peer '{remote_name}'")

        remote_expected_key = f"pair:{pair_name}/{remote_name}/expected_world_size"

        while not self.store.check([remote_expected_key]):
            time.sleep(TIME_INTERVAL)

        expected_remote = int(self.store.get(remote_expected_key).decode())
        logger.debug(f"Agent {self.rank}: Remote peer '{remote_name}' expects {expected_remote} ranks")

        # Then, wait for all remote ranks to register
        remote_ranks = self._wait_until_ranks_ready(
            key_pattern=f"pair:{pair_name}/rank:{{rank}}/{remote_name}",
            candidate_ranks=set(range(self.world_size)),
            expected_count=expected_remote,
            description=f"Waiting for remote peer '{remote_name}'",
        )

        logger.info(f"Agent {self.rank}: Remote peer '{remote_name}' complete: {remote_ranks}")
        logger.debug(f"Agent {self.rank}: Creating pair group with ranks: {local_ranks + remote_ranks}")
        pair_group = dist.new_group(ranks=(local_ranks + remote_ranks))
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
            local_group = dist.new_group(local_ranks)
            remote_group = dist.new_group(remote_ranks)
        else:
            remote_group = dist.new_group(remote_ranks)
            local_group = dist.new_group(local_ranks)

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

        # Step 10: Set transfer ready and signal flags to 0
        transfer_ready_key = f"pair:{pair_name}/rank:{self.rank}/state:ready"
        self.store.set(transfer_ready_key, "0")
        logger.debug(f"Agent {self.rank}: Set {transfer_ready_key} = '0'")
        transfer_singal_key = f"pair:{pair_name}/state:transfer_signal"
        self.store.set(transfer_singal_key, "0")
        logger.debug(f"Agent {self.rank}: Set {transfer_singal_key} = '0'")

        # Step 11: Write PairState to State LMDB (for Worker verification)
        state_key = f"pair:{pair_name}/state:match".encode()
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

        # Batch-scoped synchronization: set ready flag with batch_id
        transfer_ready_key = f"batch:{batch_id}/rank:{self.rank}/state:ready"
        self.store.set(transfer_ready_key, "1")
        logger.debug(f"Agent {self.rank}: Set {transfer_ready_key} = '1'")

        transfer_signal_key = f"batch:{batch_id}/state:transfer_signal"
        self.store.set(transfer_signal_key, "1")
        logger.debug(f"Agent {self.rank}: Set {transfer_signal_key} = '1'")

        # Collect all ranks involved in this batch (across all pairs)
        all_ranks = set()
        for pair_name in batch_state.pair_names:
            pair_state = self.pairs[pair_name]
            all_ranks.update(pair_state.local_ranks)
            all_ranks.update(pair_state.remote_ranks)

        total_ranks = len(all_ranks)
        logger.info(
            f"Agent {self.rank}: Batch {batch_id}: Waiting for {total_ranks} ranks across {len(batch_state.pair_names)} pairs"
        )

        # Wait for all ranks to be ready (batch-scoped)
        key_pattern = f"batch:{batch_id}/rank:{{rank}}/state:ready"
        self._wait_until_ranks_ready(
            key_pattern=key_pattern,
            candidate_ranks=all_ranks,
            expected_count=total_ranks,
            description=f"Batch {batch_id}",
        )
        logger.info(f"Agent {self.rank}: Batch {batch_id}: All {total_ranks} ranks ready for transfer")

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

        # Barrier on all pair_groups involved
        for pair_name in batch_state.pair_names:
            pair_state = self.pairs[pair_name]
            dist.barrier(group=pair_state.pair_group)

        # Cleanup batch-scoped flags
        self.store.set(transfer_ready_key, "0")
        self.store.set(transfer_signal_key, "0")
        logger.info(f"Agent {self.rank}: Batch {batch_id}: Transfer complete")

    def _handle_query_status(self, msg: QueryStatus):
        batch_id = msg.batch_id
        state_name = msg.state_name  # e.g. "transfer_signal"
        tcpstore_state_key = f"batch:{batch_id}/state:{state_name}"
        statedb_key = f"batch:{batch_id}/state:{state_name}".encode()

        if batch_id not in self.batches:
            logger.error(f"Agent {self.rank}: QueryStatus for unknown batch: {batch_id}")
            return

        batch_state = self.batches[batch_id]

        # Synchronize with local group of the first pair in batch
        first_pair_name = batch_state.pair_names[0]
        torch.cuda.synchronize()
        dist.barrier(group=self.pairs[first_pair_name].local_group)  # ensure all ranks read the same state

        if state_name == "transfer_signal":
            logger.debug(f"Agent {self.rank}: Query {state_name} status for batch '{batch_id}'")
            # Check if key exists in TCPStore
            if self.store.check([tcpstore_state_key]):
                state = self.store.get(tcpstore_state_key) == b"1"
            else:
                state = False
            logger.debug(f"Agent {self.rank}: Query {state_name} status for batch '{batch_id}': {state}")
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

            if self.store.check([mesh_shape_key]) and self.store.check([placements_key]):
                # Get mesh shape base64 string and decode
                mesh_shape_b64 = self.store.get(mesh_shape_key)
                mesh_shape_bytes = base64.b64decode(mesh_shape_b64)
                mesh_shape = ForkingPickler.loads(mesh_shape_bytes)

                # Get placements base64 string and decode
                placements_b64 = self.store.get(placements_key)
                placements_bytes = base64.b64decode(placements_b64)
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

    def _wait_until_ranks_ready(
        self,
        key_pattern: str,
        candidate_ranks: set[int],
        expected_count: int,
        description: str = "",
    ) -> list[int]:
        """Wait until expected number of ranks are ready with incremental scanning.

        Args:
            key_pattern: Key pattern with {rank} placeholder (e.g. "pair:{name}/rank:{rank}/{side}")
            candidate_ranks: Set of ranks to check
            expected_count: Expected number of ready ranks
            description: Optional description for logging progress

        Returns:
            Sorted list of ready ranks
        """
        known_ready: set[int] = set()
        pending = candidate_ranks.copy()

        while len(known_ready) < expected_count:
            # Only scan pending ranks
            for r in list(pending):  # Convert to list to avoid modification during iteration
                key = key_pattern.format(rank=r)
                if self.store.check([key]):
                    try:
                        if self.store.get(key) == b"1":
                            known_ready.add(r)
                            pending.remove(r)
                    except KeyError:
                        pass

            if description:
                logger.info(f"Agent {self.rank}: {description}: {len(known_ready)}/{expected_count} ranks ready")

            if len(known_ready) < expected_count:
                time.sleep(TIME_INTERVAL)

        return sorted(known_ready)

    def _update_heartbeat(self):
        """Update heartbeat timestamp in State LMDB.

        This allows Workers to verify the Agent is alive and responsive.
        Called on startup and every main loop iteration.
        """
        with self.state_env.begin(write=True, db=self.state_db) as txn:
            txn.put(b"agent:heartbeat", str(time.time()).encode())

    def _release_semaphore(self, semaphore_name: str):
        try:
            # Open the semaphore (must be created by client)
            sem = posix_ipc.Semaphore(semaphore_name)
            sem.release()
            logger.debug(f"Agent {self.rank}: Released semaphore '{semaphore_name}'")
        except posix_ipc.ExistentialError:
            logger.warning(f"Agent {self.rank}: Semaphore release '{semaphore_name}' not found for release")
        except Exception as e:
            logger.error(f"Agent {self.rank}: Error releasing semaphore '{semaphore_name}': {e}")

        try:
            sem.close()
            sem.unlink()
            logger.debug(f"Agent {self.rank}: Closed and unlinked semaphore '{semaphore_name}'")
        except posix_ipc.ExistentialError:
            logger.debug(f"Agent {self.rank}: Semaphore '{semaphore_name}' not found for close or unlink")
        except Exception as e:
            logger.error(f"Agent {self.rank}: Error closing or unlinking semaphore '{semaphore_name}': {e}")

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
        dist.destroy_process_group()
