"""Tensor Bus Daemon Process."""

import time
import logging
from datetime import timedelta

import lmdb
import msgspec
import torch.distributed as dist

from .state import PairState
from .commands import Send, Receive, RegisterPair
from .command_queue import CommandQueue

logger = logging.getLogger(__name__)


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
        """Initialize Daemon.

        Args:
            rank: Rank in the torch.distributed group
            world_size: Total number of Daemons
            tcpstore_host: TCPStore server address
            tcpstore_port: TCPStore server port
            lmdb_command_queue_path: Path to CommandQueue LMDB
            lmdb_state_path: Path to State LMDB (optional, for Worker verification)
        """
        self.rank = rank
        self.world_size = world_size

        # Initialize torch.distributed
        logger.info(f"Daemon {rank}: Initializing torch.distributed")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        # Initialize TCPStore
        logger.info(f"Daemon {rank}: Connecting to TCPStore at {tcpstore_host}:{tcpstore_port}")
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
            logger.info(f"Daemon {rank}: State LMDB initialized at {lmdb_state_path}")

        # Pair registry
        self.pairs = {}

        logger.info(f"Daemon {rank}: Initialized successfully")

    def run(self):
        """Main loop: process commands from Host."""
        logger.info(f"Daemon {self.rank}: Starting main loop")

        while True:
            if self.command_queue.size() != 0:
                msg = self.command_queue.dequeue()

                if msg is not None:
                    self._handle_command(msg)

            time.sleep(0.001)  # 1ms polling interval

    def _handle_command(self, command):
        """Dispatch command to appropriate handler."""
        if isinstance(command, RegisterPair):
            self._handle_register_pair(command)
        elif isinstance(command, Send):
            self._handle_send(command)
        elif isinstance(command, Receive):
            self._handle_receive(command)
        else:
            logger.warning(f"Daemon {self.rank}: Unknown command type: {type(command)}")

    def _handle_register_pair(self, msg: RegisterPair):
        """Handle RegisterPair command.

        Flow:
        1. Write to TCPStore: pair:{pair_name}:{side_name}:rank{rank} = "1"
        2. Write expected_world_size (if first)
        3. Poll TCPStore until my side is complete
        4. Poll TCPStore until remote side is complete
        5. Create PairState(status="matched")
        """
        pair_name = msg.pair_name
        local_name = msg.local_name
        expected_local = msg.expected_world_size
        remote_name = msg.remote_name

        logger.info(f"Daemon {self.rank}: RegisterPair pair={pair_name}, local={local_name} -> remote={remote_name}")

        # Step 1: Write local registration to TCPStore
        local_key = f"pair:{pair_name}:{local_name}:rank{self.rank}"
        self.store.set(local_key, "1")
        logger.info(f"Daemon {self.rank}: Wrote {local_key} = '1' to TCPStore")

        # Verify write
        verify = self.store.get(local_key)
        logger.debug(f"Daemon {self.rank}: Verified {local_key} = {verify}")

        # Step 2: Write expected_world_size (all ranks write the same value, idempotent)
        expected_key = f"pair:{pair_name}:{local_name}:expected_world_size"
        self.store.set(expected_key, str(expected_local))
        logger.debug(f"Daemon {self.rank}: Wrote {expected_key}={expected_local}")

        # Step 3: Poll until local peer is complete
        logger.info(
            f"Daemon {self.rank}: Waiting for local peer '{local_name}' to complete (expected={expected_local})"
        )
        local_ranks = []
        while len(local_ranks) < expected_local:
            local_ranks = self._scan_peer_ranks(pair_name, local_name)
            if len(local_ranks) < expected_local:
                logger.debug(f"Daemon {self.rank}: Local peer progress: {len(local_ranks)}/{expected_local}")
                time.sleep(0.1)

        logger.info(f"Daemon {self.rank}: Local peer complete: {local_ranks}")

        # Step 4: Poll until remote peer is complete
        logger.info(f"Daemon {self.rank}: Waiting for remote peer '{remote_name}'")

        # First, wait for remote peer to write expected_world_size
        remote_expected_key = f"pair:{pair_name}:{remote_name}:expected_world_size"
        expected_remote = None
        while expected_remote is None:
            if self.store.check([remote_expected_key]):
                expected_remote = int(self.store.get(remote_expected_key).decode())
                logger.debug(f"Daemon {self.rank}: Remote peer '{remote_name}' expects {expected_remote} ranks")
            else:
                time.sleep(0.1)

        # Then, wait for all remote ranks to register
        remote_ranks = []
        while len(remote_ranks) < expected_remote:
            remote_ranks = self._scan_peer_ranks(pair_name, remote_name)
            if len(remote_ranks) < expected_remote:
                logger.debug(f"Daemon {self.rank}: Remote peer progress: {len(remote_ranks)}/{expected_remote}")
                time.sleep(0.1)

        logger.info(f"Daemon {self.rank}: Remote peer '{remote_name}' complete: {remote_ranks}")

        # Step 5: Create PairState
        state = PairState(
            pair_name=pair_name,
            local_name=local_name,
            local_ranks=local_ranks,
            remote_name=remote_name,
            remote_ranks=remote_ranks,
            status="matched",
            created_at=time.time(),
            last_updated=time.time(),
        )
        self.pairs[pair_name] = state

        # Step 6: Write PairState to State LMDB (for Worker verification)
        if self.state_env and self.state_db:
            state_key = f"pair:{pair_name}:state".encode()
            state_bytes = msgspec.msgpack.encode(state)
            with self.state_env.begin(write=True, db=self.state_db) as txn:
                txn.put(state_key, state_bytes)
            logger.debug(f"Daemon {self.rank}: Wrote PairState to State LMDB")

        logger.info(
            f"Daemon {self.rank}: Pair '{pair_name}' matched! "
            f"Local '{local_name}': {local_ranks}, Remote '{remote_name}': {remote_ranks}"
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
        logger.debug(
            f"Daemon {self.rank}: Scanning for {pair_name}:{peer_name} (checking ranks 0-{self.world_size - 1})"
        )

        for r in range(self.world_size):
            key = f"pair:{pair_name}:{peer_name}:rank{r}"
            # Use check() instead of get() to avoid blocking
            if self.store.check([key]):
                # Key exists, now get it
                value = self.store.get(key)
                logger.debug(f"Daemon {self.rank}: Got {key} = {value}")
                if value == b"1":
                    ranks.append(r)
            else:
                logger.debug(f"Daemon {self.rank}: Key {key} not found")

        logger.debug(f"Daemon {self.rank}: _scan_peer_ranks({pair_name}, {peer_name}) = {ranks}")
        return ranks

    def _handle_send(self, msg: Send):
        """Handle Send command (future implementation)."""
        logger.warning(f"Daemon {self.rank}: Send for pair '{msg.pair_name}' not implemented yet")

    def _handle_receive(self, msg: Receive):
        """Handle Receive command (future implementation)."""
        logger.warning(f"Daemon {self.rank}: Receive for pair '{msg.pair_name}' not implemented yet")

    def close(self):
        """Cleanup resources."""
        self.command_queue.close()
        if self.state_env:
            self.state_env.close()
        dist.destroy_process_group()
