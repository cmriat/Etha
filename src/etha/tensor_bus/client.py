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
    ):
        # CommandQueue will handle its own env var fallback
        self.command_queue = CommandQueue(lmdb_command_queue_path)

        # Get state path from argument or environment variable
        if agent_state_lmdb_path is None:
            agent_state_lmdb_path = os.environ.get("TENSOR_BUS_STATE_PATH")
        if agent_state_lmdb_path is None:
            raise ValueError(
                "State LMDB path not provided. Either pass agent_state_lmdb_path argument "
                "or set TENSOR_BUS_STATE_PATH environment variable."
            )

        self.agent_state_lmdb_path = agent_state_lmdb_path
        self.handlers = {}  # pair_name -> PairHandler

        # Open state LMDB once for reuse
        self.state_env = lmdb.open(
            self.agent_state_lmdb_path,
            readonly=True,
            lock=False,
            subdir=False,
            max_dbs=2,  # Must match Agent's max_dbs
        )
        self.state_db = self.state_env.open_db(b"pair_state")

        logger.info(f"TensorBusClient: Initialized with CommandQueue at {self.command_queue.lmdb_path}")
        logger.info(f"TensorBusClient: State LMDB path: {agent_state_lmdb_path}")

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

        # Wait for Agent to process
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
