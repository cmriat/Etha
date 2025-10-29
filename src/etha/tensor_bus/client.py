"""Tensor Bus Client (Host-side).

Host processes use TensorBusClient to register pairs and communicate with Agents.
"""

import os
import time
import logging
import weakref
from typing import Literal
from multiprocessing.reduction import ForkingPickler

import lmdb
import torch
import msgspec
import posix_ipc

from .commands import Transfer, QueryStatus, RegisterPair, RegisterTensor
from .pair_state import PairState
from .command_queue import CommandQueue

logger = logging.getLogger(__name__)


def generate_semaphore_name(command_type: str, pair_name: str) -> str:
    counter = int(time.time() * 1000) % 1000
    safe_pair = pair_name.replace("/", "_").replace(":", "_")
    return f"/command_{command_type}_{safe_pair}_{counter}"


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
        self.handlers: dict[str, PairHandler] = {}

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
                logger.error(
                    f"TensorBusClient[{self.agent_rank}]: {command_type} timeout for pair '{pair_name}' with semaphore {sem_name}"
                )
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
        tensor: torch.Tensor,
        expected_world_size: int,
        blocking: bool = True,
        timeout: float = 30.0,
    ) -> "PairHandler":
        """Register a pair and return handler.

        Args:
            pair_name: Unique identifier for this pair
            local_name: Name of local peer
            remote_name: Name of remote peer
            tensor: Local tensor to be registered
            expected_world_size: Number of ranks for local peer

        Returns:
            PairHandler for this pair

        Blocks until:
            - Agent registers to TCPStore
            - Remote peer registers
            - Pair is matched
        """
        logger.info(
            f"TensorBusClient[{self.agent_rank}]: Registering pair '{pair_name}' as '{local_name}' -> '{remote_name}'"
        )

        # Create RegisterPair command
        msg = RegisterPair(
            pair_name=pair_name,
            local_name=local_name,
            expected_world_size=expected_world_size,
            remote_name=remote_name,
        )

        self._execute_command_with_semaphore(msg, "register", pair_name, blocking=blocking, timeout=timeout)

        # Get the matched state
        state_key = f"pair:{pair_name}/state:match".encode()
        with self.state_env.begin(db=self.state_db) as txn:
            state_bytes = txn.get(state_key)
            if state_bytes:
                matched_state = msgspec.msgpack.Decoder(PairState).decode(state_bytes)
                if matched_state.status != "matched":
                    raise RuntimeError(f"Pair '{pair_name}' is not matched")
                else:
                    logger.info(
                        f"TensorBusClient[{self.agent_rank}]: Pair '{pair_name}' matched! "
                        f"Local ranks: {matched_state.local_ranks}, Remote ranks: {matched_state.remote_ranks}"
                    )

        # Create PairHandler
        handler = PairHandler(client=self, pair_name=pair_name, tensor=tensor)
        self.handlers[pair_name] = handler

        logger.info(f"TensorBusClient[{self.agent_rank}]: Pair '{pair_name}' registered successfully")

        return handler

    def transfer(
        self, pair_name: str, transfer_type: Literal["send", "recv"], blocking: bool = False, timeout: float = 30.0
    ) -> posix_ipc.Semaphore:
        msg = Transfer(pair_name=pair_name, transfer_type=transfer_type)
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

    def register_tensor(
        self, pair_name: str, tensor_name: str, tensor: torch.Tensor, blocking: bool = True, timeout: float = 30.0
    ):
        msg = RegisterTensor(pair_name=pair_name, tensor_name=tensor_name, tensor_payload=ForkingPickler.dumps(tensor))
        return self._execute_command_with_semaphore(
            msg, "register_tensor", pair_name, blocking=blocking, timeout=timeout
        )

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
                logger.debug(f"TensorBusClient[{self.agent_rank}]: Heartbeat check error (retrying): {e}")

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


class PairHandler:
    """Lightweight handler for a single Pair."""

    def __init__(self, client: TensorBusClient, pair_name: str, tensor: torch.Tensor):
        """Initialize PairHandler.

        Args:
            client: TensorBusClient instance
            pair_name: Pair name
            tensor: Local tensor
        """
        self._client_ref = weakref.ref(client)  # Weak reference to avoid circular ref
        self.pair_name = pair_name
        self.tensor = tensor

    @property
    def client(self) -> TensorBusClient:
        """Get client from weak reference.

        Raises:
            RuntimeError: If client has been garbage collected
        """
        client = self._client_ref()
        if client is None:
            raise RuntimeError(
                f"TensorBusClient[{self.client.agent_rank}]: has been garbage collected. PairHandler '{self.pair_name}' is no longer valid."
            )
        return client

    def transfer(
        self, transfer_type: Literal["send", "recv"], blocking: bool = False, timeout: float = 30.0
    ) -> posix_ipc.Semaphore:
        """Transfer tensor to remote side.

        Args:
            blocking: If True, block until send completes

        Returns:
            Semaphore for operation completion
        """
        return self.client.transfer(self.pair_name, transfer_type, blocking=blocking, timeout=timeout)

    def query_transfer_signal(self, blocking: bool = True, timeout: float = 30.0) -> bool:
        """Query transfer signal status of the pair.

        Returns:
            if the transfer signal is active
        """
        return self.client.query_transfer_signal(self.pair_name, blocking=blocking, timeout=timeout)

    def register_tensor(self, tensor_name: str, tensor: torch.Tensor, blocking: bool = False, timeout: float = 30.0):
        """Register a tensor to the pair."""
        return self.client.register_tensor(self.pair_name, tensor_name, tensor, blocking=blocking, timeout=timeout)

    def close(self):
        """Cleanup (placeholder for future use)."""
        pass
