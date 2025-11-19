"""Tensor Bus Client (Host-side).

Host processes use TensorBusClient to register pairs and communicate with Agents.
"""

from __future__ import annotations

import os
import time
import uuid
import logging
import weakref
import traceback
from typing import Literal
from multiprocessing.reduction import ForkingPickler

import lmdb
import torch
import msgspec
import posix_ipc
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

from .commands import Transfer, QueryStatus, RegisterPair, RegisterTensors
from .command_queue import CommandQueue

logger = logging.getLogger(__name__)


class BatchHandler:
    """Handler for batch tensor operations across multiple pairs."""

    def __init__(self, client: TensorBusClient, pair_names: list[str]):
        """Initialize BatchHandler.

        Args:
            client: TensorBusClient instance
            pair_names: List of pair names managed by this handler
        """
        self._client_ref = weakref.ref(client)
        self.pair_names = pair_names

    @property
    def client(self) -> TensorBusClient:
        """Get client from weak reference.

        Raises:
            RuntimeError: If client has been garbage collected
        """
        client = self._client_ref()
        if client is None:
            raise RuntimeError(f"TensorBusClient has been garbage collected. BatchHandler is no longer valid.")
        return client

    def transfer(
        self, transfer_type: Literal["send", "recv"], blocking: bool = False, timeout: float = 30.0
    ) -> list[posix_ipc.Semaphore]:
        """Transfer tensors for all pairs simultaneously.

        Args:
            transfer_type: "send" or "recv"
            blocking: If True, block until all transfers complete
            timeout: Timeout in seconds

        Returns:
            List of semaphores for operation completion
        """
        semaphores = []
        for pair_name in self.pair_names:
            sem = self.client.transfer(pair_name, transfer_type, blocking=False, timeout=timeout)
            semaphores.append(sem)

        if blocking:
            for sem in semaphores:
                try:
                    sem.acquire(timeout=timeout)
                except posix_ipc.BusyError as e:
                    raise TimeoutError(f"Transfer timeout for batch handler") from e
                finally:
                    sem.close()

        return semaphores

    def query_transfer_signal(self, blocking: bool = True, timeout: float = 30.0) -> dict[str, bool]:
        """Query transfer signal status for all pairs.

        Returns:
            Dict mapping pair_name to signal status
        """
        results = {}
        for pair_name in self.pair_names:
            results[pair_name] = self.client.query_transfer_signal(pair_name, blocking=blocking, timeout=timeout)
        return results

    def close(self):
        """Cleanup (placeholder for future use)."""
        pass


def generate_semaphore_name(command_type: str, pair_name: str) -> str:
    safe_pair = pair_name.replace("/", "_").replace(":", "_")
    unique_suffix = uuid.uuid4().hex  # UUID ensures semaphore uniqueness without global state
    return f"/command_{command_type}_{safe_pair}_{os.getpid()}_{unique_suffix}"


class TensorBusClient:
    """Host-side Tensor Bus Client."""

    def __init__(
        self,
        agent_rank: int,
        lmdb_command_queue_path: str | None = None,
        agent_state_lmdb_path: str | None = None,
        connection_timeout: float = 30.0,
    ):
        """Initialize TensorBusClient.

        Args:
            lmdb_command_queue_path: Path to Agent's CommandQueue LMDB
            agent_state_lmdb_path: Path to Agent's State LMDB
            connection_timeout: Max time to wait for Agent connection (seconds)

        Raises:
            ValueError: If agent_state_lmdb_path is not provided
            ConnectionError: If Agent is not found or not responding
        """
        self.agent_rank = agent_rank
        # Step 1: Connection to agent
        if agent_state_lmdb_path is None:
            agent_state_lmdb_path = os.environ.get("TENSOR_BUS_STATE_PATH")
        if agent_state_lmdb_path is None:
            raise ValueError(
                "State LMDB path not provided. Either pass agent_state_lmdb_path argument "
                "or set TENSOR_BUS_STATE_PATH environment variable."
            )

        self.agent_state_lmdb_path = agent_state_lmdb_path

        self.state_env = None
        self.state_db = None
        self._connect_agent(agent_state_lmdb_path, timeout=connection_timeout)

        # Step 2: Initialize CommandQueue
        self.command_queue = CommandQueue(lmdb_command_queue_path)

    def _execute_command_with_semaphore(
        self, command, command_type: str, pair_name: str, blocking: bool = False, timeout: float = 30.0
    ) -> posix_ipc.Semaphore:
        sem_name = generate_semaphore_name(command_type, pair_name)
        sem = posix_ipc.Semaphore(sem_name, flags=posix_ipc.O_CREAT, initial_value=0)

        # Set semaphore name on command
        command.semaphore_name = sem_name
        command.timestamp = time.time()

        # Send command
        self.command_queue.enqueue(command)
        logger.debug(
            f"TensorBusClient[{self.agent_rank}]: Sent {command_type} command for pair '{pair_name}' with semaphore {sem_name}"
        )

        if blocking:
            try:
                sem.acquire(timeout=timeout)
                logger.debug(
                    f"TensorBusClient[{self.agent_rank}]: {command_type} completed for pair '{pair_name}' with semaphore {sem_name}"
                )
            except posix_ipc.BusyError as e:
                raise TimeoutError(
                    f"TensorBusClient[{self.agent_rank}]: {command_type} timeout for pair '{pair_name}' with semaphore {sem_name}"
                ) from e
            finally:
                sem.close()

        return sem

    def register_pair(
        self,
        pair_name: str,
        local_name: str,
        remote_name: str,
        expected_world_size: int,
        device_mesh: DeviceMesh | None = None,
        placements: tuple[Placement, ...] | None = None,
        blocking: bool = True,
        timeout: float = 30.0,
    ) -> None:
        """Register a pair for communication.

        Args:
            pair_name: Unique identifier for this pair
            local_name: Name of local peer
            remote_name: Name of remote peer
            expected_world_size: Number of ranks for local peer
            device_mesh: Local device mesh configuration
            placements: Local tensor placement strategy

        Blocks until:
            - Agent registers to TCPStore
            - Remote peer registers
            - Pair is matched
        """
        logger.info(
            f"TensorBusClient[{self.agent_rank}]: Registering pair '{pair_name}' as '{local_name}' -> '{remote_name}'"
        )

        mesh_shape_payload = None
        placements_payload = None

        if device_mesh is not None:
            mesh_shape_bytes = ForkingPickler.dumps(tuple(device_mesh.mesh.shape))
            mesh_shape_payload = memoryview(mesh_shape_bytes)

        if placements is not None:
            placements_bytes = ForkingPickler.dumps(placements)
            placements_payload = memoryview(placements_bytes)

        msg = RegisterPair(
            pair_name=pair_name,
            local_name=local_name,
            expected_world_size=expected_world_size,
            remote_name=remote_name,
            mesh_shape_payload=mesh_shape_payload,
            placements_payload=placements_payload,
        )

        self._execute_command_with_semaphore(msg, "register", pair_name, blocking=blocking, timeout=timeout)

        # Get the matched state
        state_key = f"pair:{pair_name}/state:match".encode()
        with self.state_env.begin(db=self.state_db) as txn:
            state_bytes = txn.get(state_key)
            if state_bytes:
                matched_state = msgspec.msgpack.Decoder(str).decode(state_bytes)
                if matched_state != "matched":
                    raise RuntimeError(f"Pair '{pair_name}' is not matched")
                else:
                    logger.info(f"TensorBusClient[{self.agent_rank}]: Pair '{pair_name}' matched! ")

        logger.info(f"TensorBusClient[{self.agent_rank}]: Pair '{pair_name}' registered successfully")

    def transfer(
        self, pair_name: str, transfer_type: Literal["send", "recv"], blocking: bool = False, timeout: float = 30.0
    ) -> posix_ipc.Semaphore:
        msg = Transfer(pair_name=pair_name, transfer_type=transfer_type)
        logger.info(
            f"TensorBusClient[{self.agent_rank}]: Sending transfer command for pair '{pair_name} {transfer_type}'"
        )
        return self._execute_command_with_semaphore(msg, "transfer", pair_name, blocking=blocking, timeout=timeout)

    def query_transfer_signal(self, pair_name: str, blocking: bool = True, timeout: float = 30.0) -> bool:
        query_msg = QueryStatus(pair_name=pair_name, state_name="transfer_signal")
        logger.info(f"TensorBusClient[{self.agent_rank}]: Query transfer signal status for pair '{pair_name}'")
        # Execute with semaphore synchronization (blocking)
        self._execute_command_with_semaphore(query_msg, "query", pair_name, blocking=blocking, timeout=timeout)

        # Get the status from LMDB
        state_key = f"pair:{pair_name}/state:transfer_signal".encode()
        with self.state_env.begin(db=self.state_db) as txn:
            state_bytes = txn.get(state_key)
            if state_bytes:
                state = msgspec.msgpack.Decoder(bool).decode(state_bytes)
                logger.info(
                    f"TensorBusClient[{self.agent_rank}]: Query transfer signal status for pair '{pair_name}': {state}"
                )
                return state
            else:
                return False

    def register_tensors(
        self,
        tensors: list[tuple[torch.Tensor, str]],
        bucket_size: int | None = None,
        blocking: bool = True,
        timeout: float = 30.0,
    ) -> BatchHandler:
        """Register multiple tensors across pairs.

        Args:
            tensors: list of (tensor, pair_name) tuples
            bucket_size: optional bucket size in bytes for bucketization optimization
            blocking: whether to block until completion
            timeout: timeout in seconds

        Returns:
            BatchHandler for managing the registered tensors
        """
        if not tensors:
            raise ValueError("tensors list cannot be empty")

        tensor_tuples = [(pair_name, (ForkingPickler.dumps(tensor.detach()))) for tensor, pair_name in tensors]

        msg = RegisterTensors(tensors=tensor_tuples, bucket_size=bucket_size)

        sem = self._execute_command_with_semaphore(msg, "register_tensors", "batch", blocking=blocking, timeout=timeout)

        unique_pair_names = list(dict.fromkeys(pair_name for _, pair_name in tensors))
        return BatchHandler(client=self, pair_names=unique_pair_names)

    def _connect_agent(self, path: str, timeout: float):
        """Connect to Agent.

        Args:
            path: Path to Agent's State LMDB
            timeout: Maximum time to wait (seconds)

        Raises:
            ConnectionError: Agent not found or not responding
        """
        deadline = time.time() + timeout

        # Step 1: Wait for LMDB file to exist
        logger.info(f"TensorBusClient[{self.agent_rank}]: Validating connection to Agent at {path}")
        while not os.path.exists(path):
            if time.time() > deadline:
                raise ConnectionError(
                    f"Agent LMDB not found at {path} after {timeout}s. "
                    f"Please ensure Agent process is running. "
                    f"Check AGENT_RANK environment variable is correct."
                )
            time.sleep(0.1)

        logger.debug(f"TensorBusClient[{self.agent_rank}]: Agent LMDB file found at {path}")

        # Step 2: Open LMDB
        self.state_env = lmdb.open(
            path,
            readonly=True,
            lock=False,
            subdir=False,
            max_dbs=2,
        )
        self.state_db = self.state_env.open_db(b"pair_state", create=False)

        # Step 3: Verify heartbeat is fresh
        while time.time() <= deadline:
            heartbeat_fresh = False
            heartbeat_age = None

            try:
                with self.state_env.begin(db=self.state_db) as txn:
                    heartbeat_bytes = txn.get(b"agent:heartbeat")
                    if heartbeat_bytes:
                        heartbeat_time = float(heartbeat_bytes.decode())
                        heartbeat_age = time.time() - heartbeat_time

                        if heartbeat_age < 5.0:  # Heartbeat is fresh
                            heartbeat_fresh = True

                # Validation successful - keep LMDB open and return
                if heartbeat_fresh:
                    logger.info(
                        f"TensorBusClient[{self.agent_rank}]: Agent connection validated (heartbeat age: {heartbeat_age:.2f}s)"
                    )
                    return  # Success! self.state_env remains open

            except Exception as e:
                logger.debug(
                    f"TensorBusClient[{self.agent_rank}]: Heartbeat check error (retrying): {e} {traceback.format_exc()}"
                )

            time.sleep(0.1)

        # Timeout reached - close LMDB and raise error
        self.state_env.close()
        self.state_env = None
        self.state_db = None
        raise ConnectionError(
            f"Agent at {path} is not responding after {timeout}s. "
            f"Heartbeat is too old or missing. Agent may have crashed."
        )

    def close(self):
        """Cleanup resources."""
        self.command_queue.close()
        if self.state_env is not None:
            self.state_env.close()
        logger.info(f"TensorBusClient[{self.agent_rank}]: Closed")
