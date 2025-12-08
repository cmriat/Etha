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

from .commands import InitPair, Transfer, QueryStatus, CleanupBatch, RegisterTensors
from .command_queue import CommandQueue

logger = logging.getLogger(__name__)


class BatchHandler:
    """Handler for batch tensor operations across multiple pairs."""

    def __init__(self, client: TensorBusClient, batch_id: str, pair_names: list[str]):
        """Initialize BatchHandler.

        Args:
            client: TensorBusClient instance
            batch_id: Unique identifier for this batch
            pair_names: List of pair names managed by this handler
        """
        self._client_ref = weakref.ref(client)
        self.batch_id = batch_id
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
    ) -> posix_ipc.Semaphore:
        """Transfer all tensors in this batch atomically.

        Sends a single Transfer command to execute all pairs in the batch simultaneously
        using flattened chunks/buckets for optimal performance.

        Args:
            transfer_type: "send" or "recv"
            blocking: If True, block until transfer completes
            timeout: Timeout in seconds

        Returns:
            Semaphore for operation completion
        """
        msg = Transfer(batch_id=self.batch_id, transfer_type=transfer_type)
        return self.client._execute_command_with_semaphore(
            msg, "transfer", context_id=f"batch_{self.batch_id}_{transfer_type}", blocking=blocking, timeout=timeout
        )

    def query_transfer_signal(self, blocking: bool = True, timeout: float = 30.0) -> bool:
        """Query transfer signal status for this batch.

        Returns:
            Transfer signal status (True if sender has completed transfer)
        """
        return self.client.query_transfer_signal(self.batch_id, blocking=blocking, timeout=timeout)

    def close(self, blocking: bool = True, timeout: float = 30.0):
        """Explicitly cleanup batch state in agent.

        Sends a CleanupBatch command to free resources associated with this batch.
        After calling close(), this handler should not be used.

        Args:
            blocking: If True, block until cleanup completes
            timeout: Timeout in seconds
        """
        msg = CleanupBatch(batch_id=self.batch_id)
        self.client._execute_command_with_semaphore(
            msg, "cleanup_batch", context_id=f"batch_{self.batch_id}", blocking=blocking, timeout=timeout
        )

    def __del__(self):
        """Best-effort cleanup on handler destruction."""
        try:
            # Use non-blocking cleanup in destructor to avoid hanging during shutdown
            self.close(blocking=False)
        except Exception:
            pass


def generate_semaphore_name(command_type: str, context_id: str) -> str:
    safe_context = context_id.replace("/", "_").replace(":", "_")
    unique_suffix = uuid.uuid4().hex  # UUID ensures semaphore uniqueness without global state
    return f"/command_{command_type}_{safe_context}_{os.getpid()}_{unique_suffix}"


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

        self.agent_state_lmdb_path = agent_state_lmdb_path

        self.state_env: lmdb.Environment | None = None
        self.state_db = None
        self._connect_agent(agent_state_lmdb_path, timeout=connection_timeout)

        # Step 2: Initialize CommandQueue
        if lmdb_command_queue_path is None:
            lmdb_command_queue_path = os.environ.get("TENSOR_BUS_COMMAND_QUEUE_PATH")
        self.command_queue = CommandQueue(lmdb_command_queue_path)

    def _execute_command_with_semaphore(
        self, command, command_type: str, context_id: str, blocking: bool = False, timeout: float = 30.0
    ) -> posix_ipc.Semaphore:
        sem_name = generate_semaphore_name(command_type, context_id)
        sem = posix_ipc.Semaphore(sem_name, flags=posix_ipc.O_CREAT, initial_value=0)

        # Set semaphore name on command
        command.semaphore_name = sem_name
        command.timestamp = time.time()

        # Send command
        self.command_queue.enqueue(command)
        logger.debug(
            f"TensorBusClient[{self.agent_rank}]: Sent {command_type} command for context '{context_id}' with semaphore {sem_name}"
        )

        if blocking:
            try:
                sem.acquire(timeout=timeout)
                logger.debug(
                    f"TensorBusClient[{self.agent_rank}]: {command_type} completed for context '{context_id}' with semaphore {sem_name}"
                )
            except posix_ipc.BusyError as e:
                raise TimeoutError(
                    f"TensorBusClient[{self.agent_rank}]: {command_type} timeout for context '{context_id}' with semaphore {sem_name}"
                ) from e
            finally:
                sem.close()

        return sem

    def init_pair(
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

        msg = InitPair(
            pair_name=pair_name,
            local_name=local_name,
            expected_world_size=expected_world_size,
            remote_name=remote_name,
            mesh_shape_payload=mesh_shape_payload,
            placements_payload=placements_payload,
        )

        self._execute_command_with_semaphore(msg, "register", context_id=pair_name, blocking=blocking, timeout=timeout)

        # Get the matched state
        if self.state_env is None:
            raise RuntimeError("State environment not initialized")

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
        self, batch_id: str, transfer_type: Literal["send", "recv"], blocking: bool = False, timeout: float = 30.0
    ) -> posix_ipc.Semaphore:
        msg = Transfer(batch_id=batch_id, transfer_type=transfer_type)
        logger.debug(
            f"TensorBusClient[{self.agent_rank}]: Sending transfer command for pair '{batch_id} {transfer_type}'"
        )
        return self._execute_command_with_semaphore(
            msg, "transfer", context_id=batch_id, blocking=blocking, timeout=timeout
        )

    def query_transfer_signal(self, batch_id: str, blocking: bool = True, timeout: float = 30.0) -> bool:
        query_msg = QueryStatus(batch_id=batch_id, state_name="transfer_signal")
        logger.debug(f"TensorBusClient[{self.agent_rank}]: Query transfer signal status for batch '{batch_id}'")
        # Execute with semaphore synchronization (blocking)
        self._execute_command_with_semaphore(
            query_msg, "query", context_id=f"batch_{batch_id}", blocking=blocking, timeout=timeout
        )

        if self.state_env is None:
            raise RuntimeError("State environment not initialized")

        # Get the status from LMDB
        state_key = f"batch:{batch_id}/state:transfer_signal".encode()
        with self.state_env.begin(db=self.state_db) as txn:
            state_bytes = txn.get(state_key)
            if state_bytes:
                state = msgspec.msgpack.Decoder(bool).decode(state_bytes)
                logger.debug(
                    f"TensorBusClient[{self.agent_rank}]: Query transfer signal status for batch '{batch_id}': {state}"
                )
                return state
            else:
                return False

    def register_tensors(
        self,
        batch_id: str,
        tensors: list[tuple[torch.Tensor, str]],
        bucket_size: int | None = None,
        timeout: float = 30.0,
    ) -> BatchHandler:
        """Register multiple tensors across pairs.

        This operation always blocks until registration completes,
        ensuring the returned BatchHandler is immediately usable.

        Args:
            batch_id: Unique identifier for this batch (must be same on both send/recv sides)
            tensors: list of (tensor, pair_name) tuples
            bucket_size: optional bucket size in bytes for bucketization optimization
            timeout: timeout in seconds

        Returns:
            BatchHandler for managing the registered tensors
        """
        if not tensors:
            raise ValueError("tensors list cannot be empty")

        tensor_tuples = [(pair_name, (ForkingPickler.dumps(tensor.detach()))) for tensor, pair_name in tensors]

        msg = RegisterTensors(batch_id=batch_id, tensors=tensor_tuples, bucket_size=bucket_size)

        self._execute_command_with_semaphore(
            msg, "register_tensors", context_id=f"batch_{batch_id}", blocking=True, timeout=timeout
        )

        unique_pair_names = list(dict.fromkeys(pair_name for _, pair_name in tensors))
        return BatchHandler(client=self, batch_id=batch_id, pair_names=unique_pair_names)

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

            assert self.state_env is not None  # Already initialized above

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
        if self.state_env is not None:
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
