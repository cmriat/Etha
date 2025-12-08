"""EtcdStore implementation using etcd3 library."""

import fnmatch
import logging
import threading
from typing import Any
from collections.abc import Callable

import etcd3

from .base import KVStore

logger = logging.getLogger(__name__)


class EtcdStore(KVStore):
    """KVStore implementation backed by etcd.

    Uses etcd's watch mechanism for efficient waiting.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2379,
        timeout: float | None = None,
        cleanup: bool = False,
    ):
        """Initialize EtcdStore.

        Args:
            host: etcd server host
            port: etcd server port
            timeout: Connection timeout in seconds
            cleanup: If True, delete all tensorbus/ keys on init (should only be True for rank 0)
        """
        self.host = host
        self.port = port
        self._client = etcd3.client(host=host, port=port, timeout=timeout)
        logger.info(f"EtcdStore: Connected to etcd at {host}:{port}")

        if cleanup:
            deleted = self._client.delete_prefix("tensorbus/")
            if deleted:
                logger.info(f"EtcdStore: Cleaned up {deleted} stale keys with prefix 'tensorbus/'")

    def set(self, key: str, value: str) -> None:
        """Set a key-value pair."""
        self._client.put(key, value)

    def get(self, key: str) -> bytes | None:
        """Get value for a key."""
        value, _ = self._client.get(key)
        return value

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return self.get(key) is not None

    def delete(self, key: str) -> bool:
        """Delete a key."""
        return self._client.delete(key)

    def set_bytes(self, key: str, data: bytes) -> None:
        """Store binary data directly."""
        self._client.put(key, data)

    def get_bytes(self, key: str) -> bytes | None:
        """Retrieve binary data directly."""
        return self.get(key)

    # --- Watch-based waiting ---

    def _wait_with_watch(
        self,
        watch_fn: Callable[[], int],  # returns watch_id
        check_fn: Callable[[], Any | None],  # returns result if ready, None otherwise
        on_event: Callable[[Any], Any | None],  # process event, return result if done
        timeout: float,
        error_msg: str,
    ) -> Any:
        """Generic watch-and-wait pattern.

        Args:
            watch_fn: Function to setup watch, returns watch_id
            check_fn: Function to check if condition is already met
            on_event: Callback for watch events, returns result if done
            timeout: Timeout in seconds
            error_msg: Error message for TimeoutError
        """
        done = threading.Event()
        result: list[Any] = [None]
        cancelled = threading.Event()

        def callback(event: Any) -> None:
            if cancelled.is_set() or not hasattr(event, "events"):
                return
            for e in event.events:
                if hasattr(e, "key") and hasattr(e, "value"):
                    r = on_event(e)
                    if r is not None:
                        result[0] = r
                        done.set()
                        return

        # Setup watch BEFORE initial check (avoid race)
        watch_id = watch_fn(callback)
        try:
            # Check if already satisfied
            r = check_fn()
            if r is not None:
                return r

            if done.wait(timeout=timeout):
                return result[0]
            raise TimeoutError(error_msg)
        finally:
            cancelled.set()
            self._client.cancel_watch(watch_id)

    def wait_for_key(self, key: str, timeout: float = 3600.0) -> bytes:
        """Wait for a key to exist and return its value."""

        def on_event(e: Any) -> bytes | None:
            k = e.key.decode() if isinstance(e.key, bytes) else e.key
            if k == key and e.value is not None:
                return e.value
            return None

        return self._wait_with_watch(
            watch_fn=lambda cb: self._client.add_watch_callback(key, cb),
            check_fn=lambda: self.get(key),
            on_event=on_event,
            timeout=timeout,
            error_msg=f"Timeout waiting for key: {key}",
        )

    def wait_for_keys(
        self,
        key_pattern: str,
        expected_count: int,
        value: str = "1",
        timeout: float = 3600.0,
        candidate_keys: list[str] | None = None,  # noqa: ARG002 - ignored, etcd uses prefix scan
    ) -> list[str]:
        """Wait for keys matching pattern using etcd watch."""
        prefix = key_pattern[: key_pattern.find("*")] if "*" in key_pattern else key_pattern
        value_bytes = value.encode()
        matched: set[str] = set()
        lock = threading.Lock()

        def matches(k: str, v: bytes | None) -> bool:
            return v == value_bytes and fnmatch.fnmatch(k, key_pattern)

        def on_event(e: Any) -> list[str] | None:
            k = e.key.decode() if isinstance(e.key, bytes) else e.key
            if matches(k, e.value):
                with lock:
                    matched.add(k)
                    if len(matched) >= expected_count:
                        return sorted(matched)[:expected_count]
            return None

        def check_existing() -> list[str] | None:
            for v, meta in self._client.get_prefix(prefix):
                if meta:
                    k = meta.key.decode()
                    if matches(k, v):
                        matched.add(k)
            if len(matched) >= expected_count:
                return sorted(matched)[:expected_count]
            return None

        return self._wait_with_watch(
            watch_fn=lambda cb: self._client.add_watch_prefix_callback(prefix, cb),
            check_fn=check_existing,
            on_event=on_event,
            timeout=timeout,
            error_msg=f"Timeout waiting for {expected_count} keys matching '{key_pattern}', got {len(matched)}",
        )

    def close(self, cleanup: bool = True) -> None:
        """Close the etcd client."""
        if self._client:
            if cleanup:
                deleted = self._client.delete_prefix("tensorbus/")
                logger.info(f"EtcdStore: Deleted {deleted} keys with prefix 'tensorbus/'")
            self._client.close()
            logger.info("EtcdStore: Connection closed")
