"""Tensor Bus Client (Host-side).

Host processes use TensorBusClient to register pairs and communicate with Agents.
"""

import os
import time
import logging
import weakref

import lmdb
import torch
import msgspec

from .state import PairState
from .commands import Send, Receive, RegisterPair
from .command_queue import CommandQueue

logger = logging.getLogger(__name__)


class TensorBusClient:
    """Host-side Tensor Bus Client."""

    def __init__(
        self,
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

    def register_pair(
        self,
        pair_name: str,
        local_name: str,
        remote_name: str,
        tensor: torch.Tensor,
        expected_world_size: int,
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
        logger.info(f"TensorBusClient: Registering pair '{pair_name}' as '{local_name}' -> '{remote_name}'")

        msg = RegisterPair(
            pair_name=pair_name,
            local_name=local_name,
            expected_world_size=expected_world_size,
            remote_name=remote_name,
            timestamp=time.time(),
        )
        self.command_queue.enqueue(msg)

        logger.debug(f"TensorBusClient: Waiting for pair '{pair_name}' to match")

        # TODO: fix busy wait by using a notification mechanism
        state_key = f"pair:{pair_name}:state".encode()

        matched_state = None
        while matched_state is None:
            try:
                with self.state_env.begin(db=self.state_db) as txn:
                    state_bytes = txn.get(state_key)
                    if state_bytes:
                        state = msgspec.msgpack.Decoder(PairState).decode(state_bytes)
                        if state.status == "matched":
                            matched_state = state
                            logger.info(
                                f"TensorBusClient: Pair '{pair_name}' matched! "
                                f"Local ranks: {state.local_ranks}, Remote ranks: {state.remote_ranks}"
                            )
                            break
            except Exception as e:
                logger.debug(f"TensorBusClient: State polling error (retrying): {e}")

            time.sleep(0.01)

        # Create PairHandler
        handler = PairHandler(client=self, pair_name=pair_name, tensor=tensor)
        self.handlers[pair_name] = handler

        logger.info(f"TensorBusClient: Pair '{pair_name}' registered successfully")

        return handler

    def _send(self, pair_name: str, blocking: bool = False):
        """Internal: Execute send for a pair.

        Args:
            pair_name: Pair name
            blocking: If True, block until send completes

        Returns:
            CommHandle (future)
        """
        msg = Send(pair_name=pair_name, timestamp=time.time())
        self.command_queue.enqueue(msg)

        logger.info(f"TensorBusClient: Sent Send command for pair '{pair_name}'")

        if blocking:
            # TODO: Wait for completion
            pass

        # TODO: Return CommHandle

    def _recv(self, pair_name: str, blocking: bool = False):
        """Internal: Execute recv for a pair.

        Args:
            pair_name: Pair name
            blocking: If True, block until recv completes

        Returns:
            CommHandle (future)
        """
        msg = Receive(pair_name=pair_name, timestamp=time.time())
        self.command_queue.enqueue(msg)

        logger.info(f"TensorBusClient: Sent Receive command for pair '{pair_name}'")

        if blocking:
            # TODO: Wait for completion
            pass

        # TODO: Return CommHandle

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
        logger.info(f"TensorBusClient: Validating connection to Agent at {path}")
        while not os.path.exists(path):
            if time.time() > deadline:
                raise ConnectionError(
                    f"Agent LMDB not found at {path} after {timeout}s. "
                    f"Please ensure Agent process is running. "
                    f"Check AGENT_RANK environment variable is correct."
                )
            time.sleep(0.1)

        logger.debug(f"TensorBusClient: Agent LMDB file found at {path}")

        # Step 2: Open LMDB
        self.state_env = lmdb.open(
            path,
            readonly=True,
            lock=False,
            subdir=False,
            max_dbs=2,
        )
        self.state_db = self.state_env.open_db(b"pair_state")

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
                    logger.info(f"TensorBusClient: Agent connection validated (heartbeat age: {heartbeat_age:.2f}s)")
                    return  # Success! self.state_env remains open

            except Exception as e:
                logger.debug(f"TensorBusClient: Heartbeat check error (retrying): {e}")

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
        logger.info("TensorBusClient: Closed")


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
                f"TensorBusClient has been garbage collected. PairHandler '{self.pair_name}' is no longer valid."
            )
        return client

    def send(self, blocking: bool = False):
        """Send tensor to remote side.

        Args:
            blocking: If True, block until send completes

        Returns:
            CommHandle (future)
        """
        return self.client._send(self.pair_name, blocking)

    def recv(self, blocking: bool = False):
        """Receive tensor from remote side.

        Args:
            blocking: If True, block until recv completes

        Returns:
            CommHandle (future)
        """
        return self.client._recv(self.pair_name, blocking)

    def close(self):
        """Cleanup (placeholder for future use)."""
        pass
