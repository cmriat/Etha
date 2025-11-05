"""Tensor Bus Agent Process."""

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
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

from etha.comm import get_m2m_map, m2m_communicate, bind_tensors_to_chunks

from .commands import Transfer, QueryStatus, RegisterPair, RegisterTensorBatch
from .pair_state import M2MMap, PairState
from .command_queue import CommandQueue

logger = logging.getLogger(__name__)

TIME_INTERVAL = 0.001  # 1ms


class TensorBusAgent:
    """Tensor Bus Agent.

    Responsibilities:
    1. Listen to CommandQueue for Host commands
    2. Handle RegisterPair: write to TCPStore, poll until both sides ready
    3. Handle Send/Recv: execute m2m communication
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        tcpstore_host: str,
        tcpstore_port: int,
        lmdb_command_queue_path: str,
        lmdb_state_path: str | None = None,
    ):
        """Initialize Agent.

        Args:
            rank: Rank in the torch.distributed group
            world_size: Total number of Agents
            tcpstore_host: TCPStore server address
            tcpstore_port: TCPStore server port
            lmdb_command_queue_path: Path to CommandQueue LMDB
            lmdb_state_path: Path to State LMDB (optional, for Worker verification)
        """
        self.rank = rank
        self.world_size = world_size

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
            timeout=timedelta(seconds=3600),  # 1 hour timeout
            wait_for_workers=True,
        )

        # Initialize CommandQueue (for Host communication)
        self.command_queue = CommandQueue(lmdb_command_queue_path)

        # Initialize State LMDB (for Worker verification)
        self.state_env = None
        self.state_db = None
        if lmdb_state_path:
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

        # Pair registry
        self.pairs: dict[str, PairState] = {}

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
                case RegisterPair():
                    self._handle_register_pair(command)
                case Transfer():
                    self._handle_transfer(command)
                case QueryStatus():
                    self._handle_query_status(command)
                case RegisterTensorBatch():
                    self._handle_register_tensor_batch(command)
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

    def _handle_register_pair(self, msg: RegisterPair):
        """Handle RegisterPair command.

        Flow:
        1. Write to TCPStore: pair:{pair_name}:{side_name}:rank{rank} = "1"
        2. Write expected_world_size (if first)
        3. Write device mesh and placement info to TCPStore via base64 encoding
        4. Poll TCPStore until my side is complete
        5. Poll TCPStore until remote side is complete
        6. Exchange and collect device mesh/placement info from all ranks
        7. Create PairState(status="matched") with mesh/placement info
        """
        pair_name = msg.pair_name
        local_name = msg.local_name
        expected_local = msg.expected_world_size
        remote_name = msg.remote_name

        logger.info(f"Agent {self.rank}: RegisterPair pair={pair_name}, local={local_name} -> remote={remote_name}")

        # Step 1: Write local registration to TCPStore
        local_key = f"pair:{pair_name}/rank:{self.rank}/{local_name}"
        self.store.set(local_key, "1")
        logger.debug(f"Agent {self.rank}: Wrote {local_key} = '1' to TCPStore")

        # Step 2: Write expected_world_size (all ranks write the same value, idempotent)
        expected_key = f"pair:{pair_name}/{local_name}/expected_world_size"
        self.store.set(expected_key, str(expected_local))
        logger.debug(f"Agent {self.rank}: Wrote {expected_key}={expected_local}")

        # Step 3: Write device mesh and placement info to TCPStore
        if msg.mesh_shape_payload is not None and msg.placements_payload is not None:
            # Convert memoryview to bytes, then to base64 string for TCPStore
            mesh_shape_key = f"pair:{pair_name}/rank:{self.rank}/mesh_shape"
            mesh_shape_bytes = bytes(msg.mesh_shape_payload)
            self.store.set(mesh_shape_key, base64.b64encode(mesh_shape_bytes).decode("ascii"))

            placements_key = f"pair:{pair_name}/rank:{self.rank}/placements"
            placements_bytes = bytes(msg.placements_payload)
            self.store.set(placements_key, base64.b64encode(placements_bytes).decode("ascii"))

            logger.debug(f"Agent {self.rank}: Wrote device mesh and placement info to TCPStore")

        # Step 4: Poll until local peer is complete
        logger.debug(
            f"Agent {self.rank}: Waiting for local peer '{local_name}' to complete (expected={expected_local})"
        )
        local_ranks = []
        while True:
            local_ranks = self._scan_peer_ranks(pair_name, local_name)
            if len(local_ranks) < expected_local:
                time.sleep(TIME_INTERVAL)
            else:
                break

        logger.info(f"Agent {self.rank}: Local peer complete: {local_ranks}")

        # Step 5: Poll until remote peer is complete
        logger.info(f"Agent {self.rank}: Waiting for remote peer '{remote_name}'")

        # First, wait for remote peer to write expected_world_size
        remote_expected_key = f"pair:{pair_name}/{remote_name}/expected_world_size"

        expected_remote = int(self.store.get(remote_expected_key).decode())
        logger.debug(f"Agent {self.rank}: Remote peer '{remote_name}' expects {expected_remote} ranks")

        # Then, wait for all remote ranks to register
        remote_ranks = []
        while True:
            remote_ranks = self._scan_peer_ranks(pair_name, remote_name)
            if len(remote_ranks) < expected_remote:
                time.sleep(TIME_INTERVAL)
            else:
                break

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
        if local_mesh_info and remote_mesh_info:
            # Get local mesh info (this process's mesh)
            local_mesh_shape, local_placements = local_mesh_info[0]
            local_mesh_tensor = torch.arange(
                local_ranks[0], local_ranks[0] + torch.prod(torch.tensor(local_mesh_shape))
            ).view(local_mesh_shape)
            remote_mesh_shape, remote_placements = remote_mesh_info[0]
            remote_mesh_tensor = torch.arange(
                remote_ranks[0], remote_ranks[0] + torch.prod(torch.tensor(remote_mesh_shape))
            ).view(remote_mesh_shape)
            logger.info(f"Agent {self.rank}: Local mesh: {local_mesh_tensor} with placements: {local_placements}")
            logger.info(f"Agent {self.rank}: Remote mesh: {remote_mesh_tensor} with placements: {remote_placements}")

            # Determine canonical ordering to ensure all ranks call get_m2m_map in same order
            # This prevents deadlock in collective operations within get_m2m_map
            # We use alphabetical order of role names as the tie-breaker
            local_is_first = local_name < remote_name
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

            # Generate M2M maps (topology layer, shape-independent)
            # IMPORTANT: All ranks must call in the same order to avoid deadlock
            # First call: first_mesh -> second_mesh
            forward_map_1, reverse_map_1, source_slicers_1, target_slicers_1 = get_m2m_map(
                source_mesh=first_mesh,
                source_placements=first_placements,
                target_mesh=second_mesh,
                target_placements=second_placements,
                group=pair_group,
                device="cuda",
            )
            # Second call: second_mesh -> first_mesh
            forward_map_2, reverse_map_2, source_slicers_2, target_slicers_2 = get_m2m_map(
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
                    forward_map=forward_map_1,
                    reverse_map=reverse_map_1,
                    source_num_slicers=source_slicers_1,
                    target_num_slicers=target_slicers_1,
                )
                m2m_map_recv = M2MMap(
                    forward_map=forward_map_2,
                    reverse_map=reverse_map_2,
                    source_num_slicers=source_slicers_2,
                    target_num_slicers=target_slicers_2,
                )
            else:
                m2m_map_send = M2MMap(
                    forward_map=forward_map_2,
                    reverse_map=reverse_map_2,
                    source_num_slicers=source_slicers_2,
                    target_num_slicers=target_slicers_2,
                )
                m2m_map_recv = M2MMap(
                    forward_map=forward_map_1,
                    reverse_map=reverse_map_1,
                    source_num_slicers=source_slicers_1,
                    target_num_slicers=target_slicers_1,
                )

            logger.info(f"Agent {self.rank}: Generated P2P maps for pair '{pair_name}'")
        else:
            logger.info(f"Agent {self.rank}: Skipping P2P map generation - missing or inconsistent mesh/placement info")

        # Step 9: Create PairState
        state = PairState(
            pair_name=pair_name,
            local_name=local_name,
            local_ranks=local_ranks,
            remote_name=remote_name,
            remote_ranks=remote_ranks,
            pair_size=expected_local + expected_remote,
            local_group=dist.new_group(local_ranks),
            pair_group=pair_group,
            status="matched",
            m2m_map_send=m2m_map_send,
            m2m_map_recv=m2m_map_recv,
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
        if self.state_env and self.state_db:
            state_key = f"pair:{pair_name}/state:match".encode()
            state_bytes = msgspec.msgpack.encode(state.status)
            with self.state_env.begin(write=True, db=self.state_db) as txn:
                txn.put(state_key, state_bytes)
            logger.debug(f"Agent {self.rank}: Wrote PairState to State LMDB")

        logger.info(
            f"Agent {self.rank}: Pair '{pair_name}' matched! "
            f"Local '{local_name}': {local_ranks}, Remote '{remote_name}': {remote_ranks}"
        )

    def _handle_transfer(self, msg: Transfer):
        pair_name = msg.pair_name
        transfer_type = msg.transfer_type
        logger.info(f"Agent {self.rank}: Handling transfer for pair '{pair_name}'")

        # Set transfer ready flag for this rank
        transfer_ready_key = f"pair:{pair_name}/rank:{self.rank}/state:ready"
        self.store.set(transfer_ready_key, "1")
        logger.debug(f"Agent {self.rank}: Set {transfer_ready_key} = '1'")

        transfer_singal_key = f"pair:{pair_name}/state:transfer_signal"
        self.store.set(transfer_singal_key, "1")
        logger.debug(f"Agent {self.rank}: Set {transfer_singal_key} = '1'")

        # Wait for all ranks to be ready
        logger.info(f"Agent {self.rank}: Waiting for all ranks to be ready for transfer")

        ready_ranks = []
        while True:
            ready_ranks = self._scan_peer_ranks(pair_name, "state:ready")
            if len(ready_ranks) < self.pairs[pair_name].pair_size:
                time.sleep(TIME_INTERVAL)
            else:
                break

        logger.info(f"Agent {self.rank}: All ranks ready for transfer")

        # Transfer tensors using per-tensor IR if available, otherwise fall back to simple send/recv
        pair_state = self.pairs[pair_name]

        if pair_state.tensor_irs:
            logger.info(f"Agent {self.rank}: Using optimized P2P transfer for pair '{pair_name}'")

            all_source_chunks = []
            all_target_chunks = []

            for tensor_name, tensor in pair_state.tensors.items():
                if tensor_name not in pair_state.tensor_irs:
                    logger.warning(f"Agent {self.rank}: No IR for tensor '{tensor_name}', skipping")
                    continue

                logger.debug(
                    f"Agent {self.rank}: Transferring tensor_name: '{tensor_name}' tensor: {tensor.shape} for pair '{pair_name}' using chunk IR"
                )
                send_ir, recv_ir = pair_state.tensor_irs[tensor_name]

                if transfer_type == "send":
                    source_chunks, target_chunks = send_ir
                else:
                    source_chunks, target_chunks = recv_ir

                logger.debug(
                    f"Agent {self.rank}: Binding {len(source_chunks)} source chunks, {len(target_chunks)} target chunks for tensor '{tensor_name}'"
                )

                # Bind tensor references to chunks
                bind_tensors_to_chunks(
                    source_chunks=source_chunks,
                    target_chunks=target_chunks,
                    source_tensor=tensor,
                    target_tensor=tensor,
                )

                # Accumulate chunks for batch execution
                all_source_chunks.extend(source_chunks)
                all_target_chunks.extend(target_chunks)

            # Execute all transfers in one batch
            if all_source_chunks or all_target_chunks:
                logger.info(
                    f"Agent {self.rank}: Executing batch transfer with {len(all_source_chunks)} source chunks, "
                    f"{len(all_target_chunks)} target chunks for {len(pair_state.tensors)} tensors"
                )
                m2m_communicate(
                    source_chunks=all_source_chunks,
                    target_chunks=all_target_chunks,
                )
                logger.info(f"Agent {self.rank}: Batch transfer completed for pair '{pair_name}'")
        else:
            # Fall back to simple send/recv without P2P optimization
            logger.info(
                f"Agent {self.rank}: Using simple send/recv transfer for pair '{pair_name}' (no P2P map available)"
            )
            for tensor_name, tensor in pair_state.tensors.items():
                logger.debug(
                    f"Agent {self.rank}: Transferring tensor_name: '{tensor_name}' tensor: {tensor.shape} for pair '{pair_name}' using simple send/recv"
                )
                if transfer_type == "send":
                    torch.distributed.send(tensor, pair_state.remote_ranks[pair_state.local_ranks.index(self.rank)])
                elif transfer_type == "recv":
                    torch.distributed.recv(tensor, pair_state.remote_ranks[pair_state.local_ranks.index(self.rank)])
                logger.debug(f"Agent {self.rank}: Transfered tensor_name: '{tensor_name}'")

        # Cleanup
        dist.barrier(group=pair_state.pair_group)
        self.store.set(transfer_ready_key, "0")
        self.store.set(transfer_singal_key, "0")
        logger.info(f"Agent {self.rank}: Transfer completed for pair '{pair_name}'")

    def _handle_query_status(self, msg: QueryStatus):
        pair_name = msg.pair_name
        state_name = msg.state_name  # e.g. "transfer_signal"
        tcpstore_state_key = f"pair:{pair_name}/state:{state_name}"
        statedb_key = f"pair:{pair_name}/state:{state_name}".encode()

        torch.cuda.synchronize()
        dist.barrier(group=self.pairs[pair_name].local_group)  # ensure all ranks read the same state

        if state_name == "transfer_signal":
            logger.debug(f"Agent {self.rank}: Query {state_name} status for pair '{pair_name}'")
            state = self.store.get(tcpstore_state_key) == b"1"
            logger.debug(f"Agent {self.rank}: Query {state_name} status for pair '{pair_name}': {state}")
        else:
            logger.error(f"Agent {self.rank}: Invalid state name: {state_name}")
            return

        with self.state_env.begin(write=True, db=self.state_db) as txn:
            txn.put(statedb_key, msgspec.msgpack.encode(state))

    def _handle_register_tensor_batch(self, msg: RegisterTensorBatch):
        """Handle RegisterTensorBatch command for batch tensor registration."""
        pair_name = msg.pair_name
        tensor_names = msg.tensor_names
        tensor_payloads = msg.tensor_payloads

        if pair_name not in self.pairs:
            raise ValueError(f"RegisterTensorBatch for unknown pair: {pair_name}")

        pair_state = self.pairs[pair_name]

        logger.info(f"Agent {self.rank}: Processing batch registration of {len(tensor_names)} tensors")

        # Process each tensor in the batch
        for tensor_name, tensor_payload in zip(tensor_names, tensor_payloads, strict=False):
            tensor = ForkingPickler.loads(tensor_payload)
            pair_state.tensors[tensor_name] = tensor

            # Generate IR for this tensor if p2p_map exists and IR not yet generated
            if pair_state.m2m_map_send and pair_state.m2m_map_recv and tensor_name not in pair_state.tensor_irs:
                tensor_shape = tensor.shape
                logger.debug(
                    f"Agent {self.rank}: Generating IR for tensor '{tensor_name}' with shape {tensor_shape} in pair '{pair_name}'"
                )

                # Generate send IR
                source_chunks_send, target_chunks_send = map_to_chunk_ir(
                    forward_map=pair_state.m2m_map_send.forward_map,
                    reverse_map=pair_state.m2m_map_send.reverse_map,
                    source_num_slicers=pair_state.m2m_map_send.source_num_slicers,
                    target_num_slicers=pair_state.m2m_map_send.target_num_slicers,
                    source_tensor_shape=tensor_shape,
                    target_tensor_shape=tensor_shape,
                    rank=self.rank,
                )

                # Generate recv IR
                source_chunks_recv, target_chunks_recv = map_to_chunk_ir(
                    forward_map=pair_state.m2m_map_recv.forward_map,
                    reverse_map=pair_state.m2m_map_recv.reverse_map,
                    source_num_slicers=pair_state.m2m_map_recv.source_num_slicers,
                    target_num_slicers=pair_state.m2m_map_recv.target_num_slicers,
                    source_tensor_shape=tensor_shape,
                    target_tensor_shape=tensor_shape,
                    rank=self.rank,
                )

                # Store IR
                pair_state.tensor_irs[tensor_name] = (
                    (source_chunks_send, target_chunks_send),
                    (source_chunks_recv, target_chunks_recv),
                )

                logger.debug(
                    f"Agent {self.rank}: Generated IR for tensor '{tensor_name}': "
                    f"send ({len(source_chunks_send)} src, {len(target_chunks_send)} tgt), "
                    f"recv ({len(source_chunks_recv)} src, {len(target_chunks_recv)} tgt)"
                )

        logger.info(f"Agent {self.rank}: Completed batch registration of {len(tensor_names)} tensors")

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

    def _scan_peer_ranks(self, pair_name: str, status_key: str) -> list[int]:
        """Scan TCPStore for all ranks of a given peer.

        Args:
            pair_name: Pair name
            status_key: Status key (e.g. "state:ready", "state:transfer_signal")

        Returns:
            List of ranks that have the given status
        """
        ranks = []
        for r in range(self.world_size):
            key = f"pair:{pair_name}/rank:{r}/{status_key}"
            if self.store.check([key]) and self.store.get(key) == b"1":
                ranks.append(r)
        return ranks

    def _update_heartbeat(self):
        """Update heartbeat timestamp in State LMDB.

        This allows Workers to verify the Agent is alive and responsive.
        Called on startup and every main loop iteration.
        """
        if self.state_env and self.state_db:
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

    def close(self):
        """Cleanup resources."""
        self.command_queue.close()
        if self.state_env:
            self.state_env.close()
        dist.destroy_process_group()
