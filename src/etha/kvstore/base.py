"""KV Store abstraction - designed around etcd capabilities."""

from abc import ABC, abstractmethod


class KVStore(ABC):
    """Abstract KV store interface."""

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Set a key-value pair.

        Args:
            key: The key to set
            value: The value to store (string)
        """
        ...

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Get value for a key.

        Args:
            key: The key to retrieve

        Returns:
            The value as bytes, or None if key doesn't exist
        """
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: The key to check

        Returns:
            True if key exists, False otherwise
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key.

        Args:
            key: The key to delete

        Returns:
            True if key was deleted, False if it didn't exist
        """
        ...

    @abstractmethod
    def wait_for_key(
        self,
        key: str,
        timeout: float = 3600.0,
    ) -> bytes:
        """Wait for a key to exist and return its value.

        Implementation:
        - etcd: Uses watch for efficient waiting
        - TCPStore: Uses polling

        Args:
            key: The key to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            The value as bytes

        Raises:
            TimeoutError: If timeout is reached before key exists
        """
        ...

    @abstractmethod
    def wait_for_keys(
        self,
        key_pattern: str,
        expected_count: int,
        value: str = "1",
        timeout: float = 3600.0,
        candidate_keys: list[str] | None = None,
    ) -> list[str]:
        """Wait until expected_count keys matching pattern have the specified value.

        Pattern syntax:
        - '*' matches any sequence of characters
        - Example: "pair:foo/rank:*/ready" matches "pair:foo/rank:0/ready", "pair:foo/rank:1/ready", etc.

        Implementation:
        - etcd: Uses watch + prefix query (event-driven, efficient), ignores candidate_keys
        - TCPStore: Uses polling with candidate_keys (required)

        Args:
            key_pattern: Pattern with '*' wildcard
            expected_count: Number of matching keys to wait for
            value: Expected value for matching keys (default "1")
            timeout: Maximum time to wait in seconds
            candidate_keys: List of candidate keys to check (required for TCPStore, ignored by etcd)

        Returns:
            List of matched keys

        Raises:
            TimeoutError: If timeout is reached before expected_count keys are found
        """
        ...

    @abstractmethod
    def set_bytes(self, key: str, data: bytes) -> None:
        """Store binary data.

        Implementation:
        - etcd: Stores bytes directly
        - TCPStore: Encodes to base64 (TCPStore only accepts strings)

        Args:
            key: The key to set
            data: Binary data to store
        """
        ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes | None:
        """Retrieve binary data.

        Implementation:
        - etcd: Returns bytes directly
        - TCPStore: Decodes from base64

        Args:
            key: The key to retrieve

        Returns:
            Binary data, or None if key doesn't exist
        """
        ...

    @abstractmethod
    def close(self, cleanup: bool = True) -> None:
        """Close the store and release resources.

        Args:
            cleanup: If True, delete coordination keys (etcd only)
        """
        ...
