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
        namespace: str = "default",
        component: str = "tensorbus",
    ):
        """Initialize EtcdStore.

        Args:
            host: etcd server host
            port: etcd server port
            timeout: Connection timeout in seconds
            cleanup: If True, delete all keys in this namespace on init (should only be True for rank 0)
            namespace: Namespace for key isolation
            component: Default component name
        """
        super().__init__(namespace, component)
        self.host = host
        self.port = port
        self._client = etcd3.client(host=host, port=port, timeout=timeout)
        logger.info(f"EtcdStore: Connected to etcd at {host}:{port} (namespace={namespace}, component={component})")

        if cleanup:
            # Clean up keys for this namespace (all components)
            cleanup_prefix = f"{namespace}/"
            deleted = self._client.delete_prefix(cleanup_prefix)
            if deleted:
                logger.info(f"EtcdStore: Cleaned up {deleted} stale keys with prefix '{cleanup_prefix}'")

    def set(self, key: str, value: str, *, component: str | None = None) -> None:
        """Set a key-value pair."""
        self._client.put(self._prefixed(key, component), value)

    def get(self, key: str, *, component: str | None = None) -> bytes | None:
        """Get value for a key."""
        value, _ = self._client.get(self._prefixed(key, component))
        return value

    def exists(self, key: str, *, component: str | None = None) -> bool:
        """Check if a key exists."""
        return self.get(key, component=component) is not None

    def delete(self, key: str, *, component: str | None = None) -> bool:
        """Delete a key."""
        return self._client.delete(self._prefixed(key, component))

    def set_bytes(self, key: str, data: bytes, *, component: str | None = None) -> None:
        """Store binary data directly."""
        self._client.put(self._prefixed(key, component), data)

    def get_bytes(self, key: str, *, component: str | None = None) -> bytes | None:
        """Retrieve binary data directly."""
        return self.get(key, component=component)

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

    def wait_for_key(self, key: str, timeout: float = 3600.0, *, component: str | None = None) -> bytes:
        """Wait for a key to exist and return its value."""
        prefixed_key = self._prefixed(key, component)

        def on_event(e: Any) -> bytes | None:
            k = e.key.decode() if isinstance(e.key, bytes) else e.key
            if k == prefixed_key and e.value is not None:
                return e.value
            return None

        return self._wait_with_watch(
            watch_fn=lambda cb: self._client.add_watch_callback(prefixed_key, cb),
            check_fn=lambda: self.get(key, component=component),
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
        *,
        component: str | None = None,
    ) -> list[str]:
        """Wait for keys matching pattern using etcd watch."""
        # Add namespace/component prefix to pattern
        prefixed_pattern = self._prefixed(key_pattern, component)
        prefix = prefixed_pattern[: prefixed_pattern.find("*")] if "*" in prefixed_pattern else prefixed_pattern
        value_bytes = value.encode()
        matched: set[str] = set()  # stores prefixed keys, strip on return
        lock = threading.Lock()

        def matches(k: str, v: bytes | None) -> bool:
            return v == value_bytes and fnmatch.fnmatch(k, prefixed_pattern)

        def on_event(e: Any) -> list[str] | None:
            k = e.key.decode() if isinstance(e.key, bytes) else e.key
            if matches(k, e.value):
                with lock:
                    matched.add(k)
                    if len(matched) >= expected_count:
                        return [self._strip_prefix(k) for k in sorted(matched)[:expected_count]]
            return None

        def check_existing() -> list[str] | None:
            for v, meta in self._client.get_prefix(prefix):
                if meta:
                    k = meta.key.decode()
                    if matches(k, v):
                        matched.add(k)
            if len(matched) >= expected_count:
                return [self._strip_prefix(k) for k in sorted(matched)[:expected_count]]
            return None

        return self._wait_with_watch(
            watch_fn=lambda cb: self._client.add_watch_prefix_callback(prefix, cb),
            check_fn=check_existing,
            on_event=on_event,
            timeout=timeout,
            error_msg=f"Timeout waiting for {expected_count} keys matching '{key_pattern}', got {len(matched)}",
        )

    def wait_for_value(
        self,
        key: str,
        expected: str,
        timeout: float = 3600.0,
        *,
        component: str | None = None,
    ) -> bytes:
        """Wait for a key to have a specific value."""
        prefixed_key = self._prefixed(key, component)
        expected_bytes = expected.encode()

        def on_event(e: Any) -> bytes | None:
            k = e.key.decode() if isinstance(e.key, bytes) else e.key
            if k == prefixed_key and e.value == expected_bytes:
                return e.value
            return None

        def check_fn() -> bytes | None:
            value = self.get(key, component=component)
            return value if value == expected_bytes else None

        return self._wait_with_watch(
            watch_fn=lambda cb: self._client.add_watch_callback(prefixed_key, cb),
            check_fn=check_fn,
            on_event=on_event,
            timeout=timeout,
            error_msg=f"Timeout waiting for key '{key}' to have value '{expected}'",
        )

    def close(self, cleanup: bool = True) -> None:
        """Close the etcd client."""
        if self._client:
            if cleanup:
                cleanup_prefix = f"{self.namespace}/"
                deleted = self._client.delete_prefix(cleanup_prefix)
                logger.info(f"EtcdStore: Deleted {deleted} keys with prefix '{cleanup_prefix}'")
            self._client.close()
            logger.info("EtcdStore: Connection closed")
