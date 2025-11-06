"""Test suite for LMDB-based CommandQueue."""

import os
import time
import tempfile
import threading

import pytest

from etha.tensor_bus import Transfer, QueryStatus, CommandQueue, RegisterTensorBatch
from etha.tensor_bus.command_queue import QueueFullError


class TestCommandQueue:
    """Comprehensive test suite for CommandQueue."""

    @pytest.fixture
    def temp_lmdb_path(self):
        """Create temporary LMDB file path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".lmdb") as f:
            path = f.name
        yield path
        # Best-effort cleanup (files may already be deleted by close(destroy=True))
        try:
            if os.path.exists(path):
                os.unlink(path)
            if os.path.exists(path + "-lock"):
                os.unlink(path + "-lock")
        except Exception:
            pass

    @pytest.fixture
    def queue(self, temp_lmdb_path):
        """Create queue instance with temp file."""
        q = CommandQueue(temp_lmdb_path)
        yield q
        # Close will destroy by default, which is what we want for tests
        try:
            q.close()
        except Exception:
            pass  # May already be closed

    # ==================== Basic Operations ====================

    def test_enqueue_dequeue_single(self, queue):
        """Test single message enqueue and dequeue."""
        msg = Transfer(pair_name="t1", transfer_type="send", timestamp=time.time())

        # Enqueue
        msg_id = queue.enqueue(msg)
        assert msg_id == 0  # First message has ID 0
        assert queue.size() == 1

        # Dequeue
        dequeued = queue.dequeue()
        assert isinstance(dequeued, Transfer)
        assert dequeued.pair_name == "t1"
        assert queue.size() == 0

    def test_dequeue_empty_queue(self, queue):
        """Test dequeue on empty queue returns None."""
        assert queue.is_empty()
        assert queue.dequeue() is None

    def test_fifo_order(self, queue):
        """Test FIFO ordering guarantee."""
        msgs = [Transfer(pair_name=f"t{i}", transfer_type="send", timestamp=time.time()) for i in range(5)]

        # Enqueue all
        for msg in msgs:
            queue.enqueue(msg)

        # Dequeue and verify order
        for i in range(5):
            dequeued = queue.dequeue()
            assert dequeued.pair_name == f"t{i}"

    # ==================== Message Type Handling ====================

    def test_mixed_message_types(self, queue):
        """Test mixed message types (Tagged Union)."""
        msgs = [
            Transfer(pair_name="t1", transfer_type="send", timestamp=1.0),
            QueryStatus(pair_name="t1", state_name="transfer_signal", timestamp=2.0),
            RegisterTensorBatch(pair_name="t1", tensor_names=["t1"], tensor_payloads=[b""], timestamp=3.0),
        ]

        # Enqueue mixed types
        for msg in msgs:
            queue.enqueue(msg)

        # Verify type identification
        msg1 = queue.dequeue()
        assert isinstance(msg1, Transfer)
        assert msg1.pair_name == "t1"

        msg2 = queue.dequeue()
        assert isinstance(msg2, QueryStatus)
        assert msg2.pair_name == "t1"
        assert msg2.state_name == "transfer_signal"

        msg3 = queue.dequeue()
        assert isinstance(msg3, RegisterTensorBatch)
        assert msg3.pair_name == "t1"
        assert msg3.tensor_names == ["t1"]

    # ==================== Batch Operations ====================

    def test_dequeue_batch(self, queue):
        """Test batch dequeue operation."""
        # Enqueue 10 messages
        for i in range(10):
            queue.enqueue(Transfer(pair_name=f"t{i}", transfer_type="send", timestamp=time.time()))

        # Batch dequeue 5 messages
        batch = queue.dequeue_batch(max_count=5)
        assert len(batch) == 5
        assert queue.size() == 5

        # Verify order
        for i, msg in enumerate(batch):
            assert msg.pair_name == f"t{i}"

    def test_dequeue_batch_exceeds_size(self, queue):
        """Test batch dequeue when max_count > queue size."""
        # Enqueue 3 messages
        for i in range(3):
            queue.enqueue(Transfer(pair_name=f"t{i}", transfer_type="recv", timestamp=1.0))

        # Request 10 but only 3 available
        batch = queue.dequeue_batch(max_count=10)
        assert len(batch) == 3
        assert queue.is_empty()

    def test_peek(self, queue):
        """Test peek does not modify queue."""
        msg = Transfer(pair_name="t1", transfer_type="recv", timestamp=1.0)
        queue.enqueue(msg)

        # Peek multiple times
        peeked1 = queue.peek()
        peeked2 = queue.peek()
        assert peeked1.pair_name == "t1"
        assert peeked2.pair_name == "t1"

        # Queue unchanged
        assert queue.size() == 1

        # Only dequeue removes it
        queue.dequeue()
        assert queue.is_empty()
        assert queue.peek() is None

    def test_clear(self, queue):
        """Test clearing the queue."""
        # Enqueue multiple messages
        for i in range(10):
            queue.enqueue(Transfer(pair_name=f"t{i}", transfer_type="recv", timestamp=1.0))

        assert queue.size() == 10

        # Clear
        queue.clear()
        assert queue.is_empty()
        assert queue.dequeue() is None

    def test_large_queue(self, queue):
        """Test handling large number of messages."""
        n = 1000

        # Enqueue 1000 messages
        for i in range(n):
            queue.enqueue(Transfer(pair_name=f"t{i}", transfer_type="send", timestamp=time.time()))

        assert queue.size() == n

        # Batch dequeue all
        all_msgs = []
        while not queue.is_empty():
            batch = queue.dequeue_batch(max_count=100)
            all_msgs.extend(batch)

        assert len(all_msgs) == n

        # Verify order
        for i, msg in enumerate(all_msgs):
            assert msg.pair_name == f"t{i}"

    def test_persistence(self, temp_lmdb_path):
        """Test queue survives close/reopen."""
        # First queue instance
        q1 = CommandQueue(temp_lmdb_path)
        q1.enqueue(Transfer(pair_name="t1", transfer_type="send", timestamp=1.0))
        q1.enqueue(Transfer(pair_name="t2", transfer_type="recv", timestamp=2.0))
        assert q1.size() == 2
        q1.close(destroy=False)  # Keep files for next instance

        # Second queue instance (reopen same file)
        q2 = CommandQueue(temp_lmdb_path)
        assert q2.size() == 2

        msg1 = q2.dequeue()
        assert isinstance(msg1, Transfer)
        assert msg1.pair_name == "t1"

        msg2 = q2.dequeue()
        assert isinstance(msg2, Transfer)
        assert msg2.pair_name == "t2"

        assert q2.is_empty()
        q2.close()  # Default destroy=True, cleanup

    def test_destroy(self, temp_lmdb_path):
        """Test close(destroy=True) completely removes LMDB files."""
        # Create queue and add some data
        q = CommandQueue(temp_lmdb_path)
        q.enqueue(Transfer(pair_name="t1", transfer_type="send", timestamp=1.0))
        q.enqueue(Transfer(pair_name="t2", transfer_type="recv", timestamp=2.0))
        assert q.size() == 2

        # Verify files exist
        assert os.path.exists(temp_lmdb_path), "Main DB file should exist"
        lock_file = temp_lmdb_path + "-lock"
        assert os.path.exists(lock_file), "Lock file should exist"

        # Destroy queue
        q.close(destroy=True)

        # Verify environment is closed
        assert q.env is None, "Environment should be None after destroy"

        # Verify files are deleted
        assert not os.path.exists(temp_lmdb_path), "Main DB file should be deleted"
        assert not os.path.exists(lock_file), "Lock file should be deleted"

    def test_destroy_idempotent(self, temp_lmdb_path):
        """Test close(destroy=True) can be called multiple times safely."""
        q = CommandQueue(temp_lmdb_path)
        q.enqueue(Transfer(pair_name="t1", transfer_type="send", timestamp=1.0))

        # First destroy
        q.close(destroy=True)
        assert not os.path.exists(temp_lmdb_path)

        # Second destroy (should not raise error)
        q.close(destroy=True)  # Files already gone, should be silent

    # ==================== Circular Queue Tests ====================

    def test_circular_queue_capacity(self, temp_lmdb_path):
        """Test circular queue respects capacity limit."""
        # Create queue with small capacity
        q = CommandQueue(temp_lmdb_path, capacity=5)

        # Fill to capacity
        for i in range(5):
            q.enqueue(Transfer(pair_name=f"t{i}", transfer_type="send"), block=False)

        assert q.size() == 5

        # Try to enqueue when full (non-blocking should fail)
        with pytest.raises(QueueFullError):
            q.enqueue(Transfer(pair_name="overflow", transfer_type="send"), block=False)

        q.close(destroy=True)

    def test_circular_queue_wraparound(self, temp_lmdb_path):
        """Test circular queue wraps around correctly."""
        q = CommandQueue(temp_lmdb_path, capacity=5)

        # Fill queue
        for i in range(5):
            q.enqueue(Transfer(pair_name=f"t{i}", transfer_type="send"), block=False)

        # Dequeue 2 items
        q.dequeue()
        q.dequeue()
        assert q.size() == 3

        # Now should be able to enqueue 2 more (wrapping around)
        q.enqueue(Transfer(pair_name="t5", transfer_type="send"), block=False)
        q.enqueue(Transfer(pair_name="t6", transfer_type="send"), block=False)
        assert q.size() == 5

        # Verify order is correct
        for i in range(2, 7):
            msg = q.dequeue()
            assert msg.pair_name == f"t{i}"

        q.close(destroy=True)

    def test_blocking_enqueue(self, temp_lmdb_path):
        """Test blocking enqueue waits for space."""
        q = CommandQueue(temp_lmdb_path, capacity=3)

        # Fill queue
        for i in range(3):
            q.enqueue(Transfer(pair_name=f"t{i}", transfer_type="send"), block=False)

        # Start a thread that will dequeue after 0.5s
        def dequeuer():
            time.sleep(0.5)
            q.dequeue()

        t = threading.Thread(target=dequeuer)
        t.start()

        # Blocking enqueue should wait
        start = time.time()
        q.enqueue(Transfer(pair_name="blocking", transfer_type="send"), block=True, timeout=2)
        elapsed = time.time() - start

        # Should have waited ~0.5s
        assert 0.4 < elapsed < 1.0

        t.join()
        q.close(destroy=True)

    def test_blocking_enqueue_timeout(self, temp_lmdb_path):
        """Test blocking enqueue times out correctly."""
        q = CommandQueue(temp_lmdb_path, capacity=2)

        # Fill queue
        for i in range(2):
            q.enqueue(Transfer(pair_name=f"t{i}", transfer_type="send"), block=False)

        # Blocking enqueue with timeout should raise QueueFullError
        with pytest.raises(QueueFullError):
            q.enqueue(Transfer(pair_name="timeout", transfer_type="send"), block=True, timeout=0.5)

        q.close(destroy=True)

    # ==================== New API Tests ====================

    def test_close_with_destroy_flag(self, temp_lmdb_path):
        """Test close(destroy=True) removes all resources."""
        q = CommandQueue(temp_lmdb_path)
        q.enqueue(Transfer(pair_name="t1", transfer_type="send", timestamp=1.0))

        # Verify files exist
        assert os.path.exists(temp_lmdb_path)
        assert os.path.exists(temp_lmdb_path + "-lock")

        # Close with destroy=True
        q.close(destroy=True)

        # Verify resources are removed
        assert q.env is None
        assert not os.path.exists(temp_lmdb_path)
        assert not os.path.exists(temp_lmdb_path + "-lock")

    def test_close_without_destroy(self, temp_lmdb_path):
        """Test close(destroy=False) keeps files."""
        q = CommandQueue(temp_lmdb_path)
        q.enqueue(Transfer(pair_name="t1", transfer_type="send", timestamp=1.0))

        # Close without destroy
        q.close(destroy=False)

        # Files should still exist
        assert os.path.exists(temp_lmdb_path)

        # Can reopen
        q2 = CommandQueue(temp_lmdb_path)
        assert q2.size() == 1
        q2.close()  # Default destroy=True, cleanup

    def test_multiprocess_semaphore_sharing(self, temp_lmdb_path):
        """Test semaphores are shared across multiple queue instances."""
        # First instance creates semaphores
        q1 = CommandQueue(temp_lmdb_path, capacity=5)
        q1.enqueue(Transfer(pair_name="t1", transfer_type="send"), block=False)
        assert q1.size() == 1

        # Verify files exist
        assert os.path.exists(temp_lmdb_path)

        # Second instance should reuse same semaphores
        q2 = CommandQueue(temp_lmdb_path, capacity=5)
        assert q2.size() == 1

        # Dequeue from q2
        msg = q2.dequeue()
        assert msg.pair_name == "t1"
        assert q2.size() == 0

        # q1 should also see empty queue
        assert q1.size() == 0

        # First close (destroy=False) - files should remain
        q1.close(destroy=False)
        assert os.path.exists(temp_lmdb_path), "Files should still exist after close(destroy=False)"
        assert os.path.exists(temp_lmdb_path + "-lock")

        # Last close (default destroy=True) - files should be deleted
        q2.close()
        assert not os.path.exists(temp_lmdb_path), "Files should be deleted after final close()"
        assert not os.path.exists(temp_lmdb_path + "-lock")
