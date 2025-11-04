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

from etha.comm import get_p2p_map, p2p_communicate

from .commands import Transfer, QueryStatus, RegisterPair, RegisterTensor
from .pair_state import PairState
from .command_queue import CommandQueue

logger = logging.getLogger(__name__)

TIME_INTERVAL = 0.001  # 1ms


class TensorBusAgent:
    """Tensor Bus Agent.

    Responsibilities:
    1. Listen to CommandQueue for Host commands
    2. Handle RegisterPair: write to TCPStore, poll until both sides ready
    3. Handle Send/Recv: execute p2p communication (future)
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
                case RegisterTensor():
                    self._handle_register_tensor(command)
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
        local_key = f"pair:{pair_name}/{local_name}/rank:{self.rank}"
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
        while len(local_ranks) < expected_local:
            local_ranks = self._scan_peer_ranks(pair_name, local_name)
            if len(local_ranks) < expected_local:
                time.sleep(TIME_INTERVAL)

        logger.info(f"Agent {self.rank}: Local peer complete: {local_ranks}")

        # Step 5: Poll until remote peer is complete
        logger.info(f"Agent {self.rank}: Waiting for remote peer '{remote_name}'")

        # First, wait for remote peer to write expected_world_size
        remote_expected_key = f"pair:{pair_name}/{remote_name}/expected_world_size"

        expected_remote = int(self.store.get(remote_expected_key).decode())
        logger.debug(f"Agent {self.rank}: Remote peer '{remote_name}' expects {expected_remote} ranks")

        # Then, wait for all remote ranks to register
        remote_ranks = []
        while len(remote_ranks) < expected_remote:
            remote_ranks = self._scan_peer_ranks(pair_name, remote_name)
            if len(remote_ranks) < expected_remote:
                time.sleep(TIME_INTERVAL)

        logger.info(f"Agent {self.rank}: Remote peer '{remote_name}' complete: {remote_ranks}")

        # Step 6: Collect device mesh and placement info from all ranks
        local_mesh_info = self._collect_mesh_placement_info(pair_name, local_ranks)
        remote_mesh_info = self._collect_mesh_placement_info(pair_name, remote_ranks)

        # Step 7: Validate mesh/placement consistency
        if local_mesh_info:
            self._validate_mesh_placement_consistency(local_mesh_info)

        if remote_mesh_info:
            self._validate_mesh_placement_consistency(remote_mesh_info)

        # Step 8: Generate P2P map if validation passed
        p2p_map_send = None
        p2p_map_recv = None
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

            # Rule: alphabetically smaller role is source first, larger is target
            send_first = local_name < remote_name
            if send_first:
                local_mesh = DeviceMesh("cuda", local_mesh_tensor)
                remote_mesh = DeviceMesh("cuda", remote_mesh_tensor)
            else:
                local_mesh = DeviceMesh("cuda", remote_mesh_tensor)
                remote_mesh = DeviceMesh("cuda", local_mesh_tensor)
                local_placements, remote_placements = remote_placements, local_placements
            logger.info(f"Agent {self.rank}: Generating P2P map for pair '{pair_name}'")

            forward_map_send, reverse_map_send, source_num_slicers_send, target_num_slicers_send = get_p2p_map(
                source_mesh=local_mesh,
                source_placements=local_placements,
                target_mesh=remote_mesh,
                target_placements=remote_placements,
                device="cuda",
            )
            forward_map_recv, reverse_map_recv, source_num_slicers_recv, target_num_slicers_recv = get_p2p_map(
                source_mesh=remote_mesh,
                source_placements=remote_placements,
                target_mesh=local_mesh,
                target_placements=local_placements,
                device="cuda",
            )
            if send_first:
                p2p_map_send = {
                    "forward_map": forward_map_send,
                    "reverse_map": reverse_map_send,
                    "source_num_slicers": source_num_slicers_send,
                    "target_num_slicers": target_num_slicers_send,
                }
                p2p_map_recv = {
                    "forward_map": forward_map_recv,
                    "reverse_map": reverse_map_recv,
                    "source_num_slicers": source_num_slicers_recv,
                    "target_num_slicers": target_num_slicers_recv,
                }
            else:
                p2p_map_send = {
                    "forward_map": forward_map_recv,
                    "reverse_map": reverse_map_recv,
                    "source_num_slicers": source_num_slicers_recv,
                    "target_num_slicers": target_num_slicers_recv,
                }
                p2p_map_recv = {
                    "forward_map": forward_map_send,
                    "reverse_map": reverse_map_send,
                    "source_num_slicers": source_num_slicers_send,
                    "target_num_slicers": target_num_slicers_send,
                }

            logger.info(f"Agent {self.rank}: Generated P2P map for pair '{pair_name}'")
            logger.info(
                f"Agent {self.rank}: Forward map: {p2p_map_send['forward_map']} source_num_slicers: {p2p_map_send['source_num_slicers']} target_num_slicers: {p2p_map_send['target_num_slicers']}"
            )
        else:
            logger.info(f"Agent {self.rank}: Skipping P2P map generation - missing or inconsistent mesh/placement info")

        # Step 9: Create PairState
        state = PairState(
            pair_name=pair_name,
            local_name=local_name,
            local_ranks=local_ranks,
            remote_name=remote_name,
            remote_ranks=remote_ranks,
            status="matched",
            created_at=time.time(),
            last_updated=time.time(),
            p2p_map_send=p2p_map_send,
            p2p_map_recv=p2p_map_recv,
        )
        self.pairs[pair_name] = state

        # Step 10: Write PairState to State LMDB (for Worker verification)
        if self.state_env and self.state_db:
            state_key = f"pair:{pair_name}/state:match".encode()
            state_bytes = msgspec.msgpack.encode(state)
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

        # Wait for all ranks to be ready
        logger.info(f"Agent {self.rank}: Waiting for all ranks to be ready for transfer")

        ready_count = 0
        ranks = self.pairs[pair_name].local_ranks + self.pairs[pair_name].remote_ranks
        while ready_count < len(ranks):
            ready_count = 0
            for rank in ranks:
                wait_key = f"pair:{pair_name}/rank:{rank}/state:ready"
                if self.store.check([wait_key]) and self.store.get(wait_key) == b"1":
                    ready_count += 1

            if ready_count < len(ranks):
                time.sleep(TIME_INTERVAL)

        logger.info(f"Agent {self.rank}: All ranks ready for transfer")

        # Transfer tensors using P2P map if available, otherwise fall back to simple send/recv
        pair_state = self.pairs[pair_name]

        if pair_state.p2p_map_send and pair_state.p2p_map_recv:
            # Use optimized P2P transfer with device mesh and placement
            logger.info(f"Agent {self.rank}: Using optimized P2P transfer for pair '{pair_name}'")

            for tensor_name, tensor in pair_state.tensors.items():
                logger.info(
                    f"Agent {self.rank}: Transferring tensor_name: '{tensor_name}' tensor: {tensor.shape} for pair '{pair_name}' using P2P map"
                )
                if transfer_type == "send":
                    forward_map = pair_state.p2p_map_send["forward_map"]
                    reverse_map = pair_state.p2p_map_send["reverse_map"]
                    source_num_slicers = pair_state.p2p_map_send["source_num_slicers"]
                    target_num_slicers = pair_state.p2p_map_send["target_num_slicers"]
                else:
                    forward_map = pair_state.p2p_map_recv["forward_map"]
                    reverse_map = pair_state.p2p_map_recv["reverse_map"]
                    source_num_slicers = pair_state.p2p_map_recv["source_num_slicers"]
                    target_num_slicers = pair_state.p2p_map_recv["target_num_slicers"]
                logger.debug(
                    f"Agent {self.rank}: Forward map: {forward_map}  Reverse map: {reverse_map} source_num_slicers: {source_num_slicers} target_num_slicers: {target_num_slicers}"
                )
                p2p_communicate(
                    source_local_tensor=tensor,
                    target_local_tensor=tensor,
                    forward_map=forward_map,
                    reverse_map=reverse_map,
                    source_num_slicers=source_num_slicers,
                    target_num_slicers=target_num_slicers,
                )
                logger.info(f"Agent {self.rank}: Transfered tensor_name: '{tensor_name}'")
        else:
            # Fall back to simple send/recv without P2P optimization
            logger.info(
                f"Agent {self.rank}: Using simple send/recv transfer for pair '{pair_name}' (no P2P map available)"
            )
            for tensor_name, tensor in pair_state.tensors.items():
                logger.info(
                    f"Agent {self.rank}: Transferring tensor_name: '{tensor_name}' tensor: {tensor.shape} for pair '{pair_name}' using simple send/recv"
                )
                for rank in pair_state.remote_ranks:
                    if transfer_type == "send":
                        torch.distributed.send(tensor, rank)
                    elif transfer_type == "recv":
                        torch.distributed.recv(tensor, rank)
                logger.info(f"Agent {self.rank}: Transfered tensor_name: '{tensor_name}'")

        # Cleanup
        dist.barrier()
        self.store.delete_key(transfer_ready_key)
        transfer_singal_key = f"pair:{pair_name}/state:transfer_signal"
        self.store.delete_key(transfer_singal_key)
        logger.info(f"Agent {self.rank}: Transfer completed for pair '{pair_name}'")

    def _handle_query_status(self, msg: QueryStatus):
        pair_name = msg.pair_name
        state_name = msg.state_name  # e.g. "transfer_signal"
        tcpstore_state_key = f"pair:{pair_name}/state:{state_name}"
        statedb_key = f"pair:{pair_name}/state:{state_name}".encode()

        if state_name == "transfer_signal":
            logger.debug(f"Agent {self.rank}: Query {state_name} status for pair '{pair_name}'")
            transfer_signal = self.store.check([tcpstore_state_key]) and self.store.get(tcpstore_state_key) == b"1"

        else:
            logger.error(f"Agent {self.rank}: Invalid state name: {state_name}")
            return

        with self.state_env.begin(write=True, db=self.state_db) as txn:
            txn.put(statedb_key, msgspec.msgpack.encode(transfer_signal))

    def _handle_register_tensor(self, msg: RegisterTensor):
        pair_name = msg.pair_name
        tensor_name = msg.tensor_name
        tensor = ForkingPickler.loads(msg.tensor_payload)

        if pair_name not in self.pairs:
            raise ValueError(f"RegisterTensor for unknown pair: {pair_name}")

        self.pairs[pair_name].tensors[tensor_name] = tensor

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

        for i, (mesh_shape, placements) in enumerate(mesh_info_list):
            # Check mesh shape consistency
            assert mesh_shape == mesh_info_list[0][0], (
                f"Agent {self.rank}: rank {i} mesh shape {mesh_shape} != reference {mesh_info_list[0][0]}"
            )

            assert placements == mesh_info_list[0][1], (
                f"Agent {self.rank}: rank {i} placements {placements} != reference {mesh_info_list[0][1]}"
            )

    def _scan_peer_ranks(self, pair_name: str, peer_name: str) -> list[int]:
        """Scan TCPStore for all ranks of a given peer.

        Args:
            pair_name: Pair name
            peer_name: Peer name (local or remote)

        Returns:
            List of ranks that have registered
        """
        ranks = []
        for r in range(self.world_size):
            key = f"pair:{pair_name}/{peer_name}/rank:{r}"
            # Use check() instead of get() to avoid blocking
            if self.store.check([key]):
                value = self.store.get(key)
                if value == b"1":
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
