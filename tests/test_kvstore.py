"""Tests for KVStore implementations (EtcdStore, TorchTCPStore)."""

import os
import time
import socket
import threading

import pytest

# Set environment variables before any imports that use network
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

from etha.kvstore import EtcdStore, TorchTCPStore


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def etcd_available() -> bool:
    """Check if etcd server is running."""
    try:
        import etcd3

        client = etcd3.client(host="localhost", port=2379, timeout=1)
        client.status()
        client.close()
        return True
    except Exception:
        return False


@pytest.fixture(params=["tcp", "etcd"])
def store(request):
    """Parametrized fixture - runs each test with both stores."""
    if request.param == "tcp":
        port = find_free_port()
        s = TorchTCPStore(
            host="localhost",
            port=port,
            world_size=1,
            is_master=True,
            timeout=60.0,
            wait_for_workers=False,
        )
        yield s
        s.close()
    else:
        if not etcd_available():
            pytest.skip("etcd not running")
        s = EtcdStore(host="localhost", port=2379)
        yield s
        s.close()


class TestKVStore:
    """Tests for KVStore interface - runs with both implementations."""

    def test_set_get(self, store):
        """Test set and get basic operation."""
        store.set("test/key1", "value1")
        result = store.get("test/key1")
        assert result == b"value1"

    def test_get_nonexistent(self, store):
        """Test get returns None for nonexistent key."""
        result = store.get("nonexistent/key")
        assert result is None

    def test_exists_true(self, store):
        """Test exists returns True for existing key."""
        store.set("test/exists", "1")
        assert store.exists("test/exists") is True

    def test_exists_false(self, store):
        """Test exists returns False for nonexistent key."""
        assert store.exists("nonexistent/key") is False

    def test_delete_existing(self, store):
        """Test delete returns True and removes the key."""
        store.set("test/delete", "1")
        assert store.exists("test/delete") is True
        result = store.delete("test/delete")
        assert result is True
        # After delete, key should not exist (or be empty for TCPStore)
        if isinstance(store, TorchTCPStore):
            # TCPStore sets to empty string instead of deleting
            assert store.get("test/delete") == b""
        else:
            assert store.exists("test/delete") is False

    def test_delete_nonexistent(self, store):
        """Test delete returns False for nonexistent key."""
        result = store.delete("nonexistent/delete")
        assert result is False

    def test_set_get_bytes(self, store):
        """Test set_bytes and get_bytes for binary data."""
        # Test with pickle-like binary data
        import pickle

        data = {"key": "value", "numbers": [1, 2, 3]}
        pickled = pickle.dumps(data)

        store.set_bytes("test/bytes", pickled)
        result = store.get_bytes("test/bytes")

        assert result == pickled
        assert pickle.loads(result) == data

    def test_get_bytes_nonexistent(self, store):
        """Test get_bytes returns None for nonexistent key."""
        result = store.get_bytes("nonexistent/bytes")
        assert result is None


class TestWaitForKeys:
    """Tests for wait_for_keys - separate class to handle store differences."""

    @pytest.fixture
    def etcd_store(self):
        """Create EtcdStore if available."""
        if not etcd_available():
            pytest.skip("etcd not running")
        s = EtcdStore(host="localhost", port=2379)
        yield s
        s.close()

    @pytest.fixture
    def tcp_store(self):
        """Create TCPStore."""
        port = find_free_port()
        s = TorchTCPStore(
            host="localhost",
            port=port,
            world_size=1,
            is_master=True,
            timeout=60.0,
            wait_for_workers=False,
        )
        yield s
        s.close()

    # --- EtcdStore tests (no candidate_keys needed) ---

    def test_etcd_wait_for_keys_existing(self, etcd_store):
        """Test wait_for_keys returns immediately when keys exist."""
        for i in range(4):
            etcd_store.set(f"test/wait/rank:{i}/ready", "1")

        result = etcd_store.wait_for_keys("test/wait/rank:*/ready", 4, timeout=5.0)
        assert len(result) == 4

    def test_etcd_wait_for_keys_async(self, etcd_store):
        """Test wait_for_keys detects keys written asynchronously."""
        pattern = "test/async/rank:*/ready"

        def writer():
            time.sleep(0.1)
            for i in range(4):
                etcd_store.set(f"test/async/rank:{i}/ready", "1")
                time.sleep(0.01)

        thread = threading.Thread(target=writer)
        thread.start()

        result = etcd_store.wait_for_keys(pattern, 4, timeout=5.0)
        thread.join()

        assert len(result) == 4

    def test_etcd_wait_for_keys_timeout(self, etcd_store):
        """Test wait_for_keys raises TimeoutError when timeout reached."""
        with pytest.raises(TimeoutError):
            etcd_store.wait_for_keys("test/timeout/rank:*/ready", 4, timeout=0.1)

    def test_etcd_wait_for_keys_pattern(self, etcd_store):
        """Test wait_for_keys correctly matches wildcard patterns."""
        etcd_store.set("test/pattern/rank:0/ready", "1")
        etcd_store.set("test/pattern/rank:1/ready", "1")
        etcd_store.set("test/pattern/rank:2/other", "1")  # Different suffix
        etcd_store.set("test/other/rank:3/ready", "1")  # Different prefix

        result = etcd_store.wait_for_keys("test/pattern/rank:*/ready", 2, timeout=1.0)

        assert len(result) == 2
        assert "test/pattern/rank:0/ready" in result
        assert "test/pattern/rank:1/ready" in result

    def test_etcd_wait_for_keys_value(self, etcd_store):
        """Test wait_for_keys only matches keys with correct value."""
        etcd_store.set("test/value/rank:0/ready", "1")
        etcd_store.set("test/value/rank:1/ready", "0")  # Wrong value
        etcd_store.set("test/value/rank:2/ready", "1")
        etcd_store.set("test/value/rank:3/ready", "1")

        result = etcd_store.wait_for_keys("test/value/rank:*/ready", 3, value="1", timeout=1.0)
        assert len(result) == 3
        assert "test/value/rank:1/ready" not in result

    # --- TorchTCPStore tests (candidate_keys required) ---

    def test_tcp_wait_for_keys_existing(self, tcp_store):
        """Test wait_for_keys returns immediately when keys exist."""
        keys = [f"test/wait/rank:{i}/ready" for i in range(4)]
        for key in keys:
            tcp_store.set(key, "1")

        result = tcp_store.wait_for_keys("test/wait/rank:*/ready", 4, candidate_keys=keys, timeout=5.0)
        assert len(result) == 4

    def test_tcp_wait_for_keys_async(self, tcp_store):
        """Test wait_for_keys detects keys written asynchronously."""
        keys = [f"test/async/rank:{i}/ready" for i in range(4)]

        def writer():
            time.sleep(0.1)
            for key in keys:
                tcp_store.set(key, "1")
                time.sleep(0.01)

        thread = threading.Thread(target=writer)
        thread.start()

        result = tcp_store.wait_for_keys("test/async/rank:*/ready", 4, candidate_keys=keys, timeout=5.0)
        thread.join()

        assert len(result) == 4

    def test_tcp_wait_for_keys_timeout(self, tcp_store):
        """Test wait_for_keys raises TimeoutError when timeout reached."""
        keys = [f"test/timeout/rank:{i}/ready" for i in range(4)]
        with pytest.raises(TimeoutError):
            tcp_store.wait_for_keys("test/timeout/rank:*/ready", 4, candidate_keys=keys, timeout=0.1)

    def test_tcp_wait_for_keys_requires_candidate_keys(self, tcp_store):
        """Test that TCPStore raises ValueError without candidate_keys."""
        with pytest.raises(ValueError, match="candidate_keys"):
            tcp_store.wait_for_keys("test/*/ready", 4)

    def test_tcp_wait_for_keys_pattern(self, tcp_store):
        """Test wait_for_keys correctly matches wildcard patterns."""
        tcp_store.set("test/pattern/rank:0/ready", "1")
        tcp_store.set("test/pattern/rank:1/ready", "1")

        candidates = [
            "test/pattern/rank:0/ready",
            "test/pattern/rank:1/ready",
            "test/pattern/rank:2/ready",  # Not written
        ]

        result = tcp_store.wait_for_keys("test/pattern/rank:*/ready", 2, candidate_keys=candidates, timeout=1.0)

        assert len(result) == 2
        assert "test/pattern/rank:0/ready" in result
        assert "test/pattern/rank:1/ready" in result

    def test_tcp_wait_for_keys_value(self, tcp_store):
        """Test wait_for_keys only matches keys with correct value."""
        keys = [f"test/value/rank:{i}/ready" for i in range(4)]
        tcp_store.set(keys[0], "1")
        tcp_store.set(keys[1], "0")  # Wrong value
        tcp_store.set(keys[2], "1")
        tcp_store.set(keys[3], "1")

        result = tcp_store.wait_for_keys("test/value/rank:*/ready", 3, value="1", candidate_keys=keys, timeout=1.0)
        assert len(result) == 3
        assert keys[1] not in result


class TestWaitForKey:
    """Tests for wait_for_key - single key waiting."""

    def test_wait_for_key_existing(self, store):
        """Test wait_for_key returns immediately when key exists."""
        store.set("test/wait_for_key/existing", "hello")
        result = store.wait_for_key("test/wait_for_key/existing", timeout=1.0)
        assert result == b"hello"

    def test_wait_for_key_async(self, store):
        """Test wait_for_key detects key written asynchronously."""
        key = "test/wait_for_key/async"

        def writer():
            time.sleep(0.1)
            store.set(key, "async_value")

        thread = threading.Thread(target=writer)
        thread.start()

        result = store.wait_for_key(key, timeout=5.0)
        thread.join()

        assert result == b"async_value"

    def test_wait_for_key_timeout(self, store):
        """Test wait_for_key raises TimeoutError when timeout reached."""
        with pytest.raises(TimeoutError):
            store.wait_for_key("test/wait_for_key/nonexistent", timeout=0.1)

    def test_wait_for_key_bytes_value(self, store):
        """Test wait_for_key returns correct bytes value."""
        store.set("test/wait_for_key/bytes", "12345")
        result = store.wait_for_key("test/wait_for_key/bytes", timeout=1.0)
        assert result == b"12345"
        assert int(result.decode()) == 12345
