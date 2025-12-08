"""TorchTCPStore implementation wrapping torch.distributed.TCPStore."""

import time
import base64
import fnmatch
import logging
from datetime import timedelta

import torch.distributed as dist

from .base import KVStore

logger = logging.getLogger(__name__)

# Polling interval for wait_for_keys
POLL_INTERVAL = 0.001


class TorchTCPStore(KVStore):
    """KVStore implementation backed by torch.distributed.TCPStore.

    This is a fallback implementation when etcd is not available.
    Uses polling for wait_for_keys (less efficient than etcd's watch).
    """

    def __init__(
        self,
        host: str,
        port: int,
        world_size: int,
        is_master: bool,
        timeout: float = 3600.0,
        wait_for_workers: bool = True,
        namespace: str = "default",
    ):
        """Initialize TorchTCPStore.

        Args:
            host: TCPStore server host
            port: TCPStore server port
            world_size: Total number of processes
            is_master: Whether this process is the master (server)
            timeout: Connection timeout in seconds
            wait_for_workers: Whether master should wait for all workers
            namespace: Namespace prefix to isolate different TensorBus instances
        """
        self.host = host
        self.port = port
        self.world_size = world_size
        self.is_master = is_master
        self.namespace = namespace
        self._key_prefix = f"tensorbus/{namespace}/"

        self._store = dist.TCPStore(
            host_name=host,
            port=port,
            world_size=world_size,
            is_master=is_master,
            timeout=timedelta(seconds=timeout),
            wait_for_workers=wait_for_workers,
        )
        logger.info(
            f"TorchTCPStore: Connected to TCPStore at {host}:{port} (master={is_master}, world_size={world_size}, namespace={namespace})"
        )

        # Track all keys we've set (for pattern matching in wait_for_keys)
        # TCPStore doesn't support prefix listing, so we need this workaround
        self._known_keys: set[str] = set()

    def set(self, key: str, value: str) -> None:
        """Set a key-value pair."""
        prefixed = self._prefixed(key)
        self._store.set(prefixed, value)
        self._known_keys.add(key)

    def get(self, key: str) -> bytes | None:
        """Get value for a key."""
        prefixed = self._prefixed(key)
        if not self._store.check([prefixed]):
            return None
        return self._store.get(prefixed)

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return self._store.check([self._prefixed(key)])

    def delete(self, key: str) -> bool:
        """Delete a key.

        Note: TCPStore doesn't support delete, so we set value to empty string.
        """
        prefixed = self._prefixed(key)
        if not self._store.check([prefixed]):
            return False
        self._store.set(prefixed, "")
        self._known_keys.discard(key)
        return True

    def wait_for_key(self, key: str, timeout: float = 3600.0) -> bytes:
        """Wait for a key to exist and return its value.

        Uses polling since TCPStore doesn't support watch.
        """
        prefixed = self._prefixed(key)
        start_time = time.monotonic()

        while True:
            if self._store.check([prefixed]):
                value = self._store.get(prefixed)
                if value:  # not empty (deleted)
                    return value

            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Timeout waiting for key: {key}")

            time.sleep(POLL_INTERVAL)

    def wait_for_keys(
        self,
        key_pattern: str,
        expected_count: int,
        value: str = "1",
        timeout: float = 3600.0,
        candidate_keys: list[str] | None = None,
    ) -> list[str]:
        """Wait for keys matching pattern using polling.

        Since TCPStore doesn't support prefix listing or watch,
        we need candidate_keys to know which keys to check.

        Args:
            key_pattern: Pattern with '*' wildcard
            expected_count: Number of matching keys to wait for
            value: Expected value for matching keys
            timeout: Maximum time to wait in seconds
            candidate_keys: List of keys to check (required for TCPStore)

        Returns:
            List of matched keys (without namespace prefix)

        Raises:
            TimeoutError: If timeout reached
            ValueError: If candidate_keys not provided
        """
        if candidate_keys is None:
            raise ValueError(
                "TorchTCPStore.wait_for_keys requires candidate_keys parameter. "
                "TCPStore doesn't support prefix listing."
            )

        value_bytes = value.encode()
        matched_keys: set[str] = set()
        start_time = time.monotonic()

        # Filter candidates by pattern first (candidates are original keys without prefix)
        candidates = [k for k in candidate_keys if fnmatch.fnmatch(k, key_pattern)]

        while len(matched_keys) < expected_count:
            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Timeout waiting for {expected_count} keys matching '{key_pattern}', got {len(matched_keys)}"
                )

            # Poll candidates (check with prefix, store without prefix)
            for key in candidates:
                if key in matched_keys:
                    continue
                prefixed = self._prefixed(key)
                if self._store.check([prefixed]):
                    try:
                        if self._store.get(prefixed) == value_bytes:
                            matched_keys.add(key)
                    except Exception:
                        pass

            if len(matched_keys) < expected_count:
                time.sleep(POLL_INTERVAL)

        return sorted(matched_keys)[:expected_count]

    def set_bytes(self, key: str, data: bytes) -> None:
        """Store binary data with base64 encoding (TCPStore only accepts strings)."""
        prefixed = self._prefixed(key)
        encoded = base64.b64encode(data).decode("ascii")
        self._store.set(prefixed, encoded)
        self._known_keys.add(key)

    def get_bytes(self, key: str) -> bytes | None:
        """Retrieve binary data with base64 decoding."""
        prefixed = self._prefixed(key)
        if not self._store.check([prefixed]):
            return None
        value = self._store.get(prefixed)
        return base64.b64decode(value)

    def close(self, cleanup: bool = True) -> None:  # noqa: ARG002
        """Close the store.

        Note: TCPStore doesn't have an explicit close method.

        Args:
            cleanup: Ignored for TCPStore (no persistent storage)
        """
        logger.info("TorchTCPStore: Connection closed")
