"""Test suite for LMDB-based CommandQueue."""

import os
import time
import tempfile

import pytest

from etha.tensor_bus.commands import Send, Receive, Register
from etha.tensor_bus.command_queue import CommandQueue


class TestCommandQueue:
    """Comprehensive test suite for CommandQueue."""

    @pytest.fixture
    def temp_lmdb_path(self):
        """Create temporary LMDB file path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".lmdb") as f:
            path = f.name
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)
        if os.path.exists(path + "-lock"):
            os.unlink(path + "-lock")

    @pytest.fixture
    def queue(self, temp_lmdb_path):
        """Create queue instance with temp file."""
        q = CommandQueue(temp_lmdb_path)
        yield q
        q.close()

    # ==================== Basic Operations ====================

    def test_enqueue_dequeue_single(self, queue):
        """Test single message enqueue and dequeue."""
        msg = Send(pair_name="t1", timestamp=time.time())

        # Enqueue
        msg_id = queue.enqueue(msg)
        assert msg_id == 0  # First message has ID 0
        assert queue.size() == 1

        # Dequeue
        dequeued = queue.dequeue()
        assert isinstance(dequeued, Send)
        assert dequeued.pair_name == "t1"
        assert queue.size() == 0

    def test_dequeue_empty_queue(self, queue):
        """Test dequeue on empty queue returns None."""
        assert queue.is_empty()
        assert queue.dequeue() is None

    def test_fifo_order(self, queue):
        """Test FIFO ordering guarantee."""
        msgs = [Send(pair_name=f"t{i}", timestamp=time.time()) for i in range(5)]

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
            Send(pair_name="t1", timestamp=1.0),
            Receive(pair_name="t1", timestamp=2.0),
            Register(handler_id="h1", timestamp=3.0),
        ]

        # Enqueue mixed types
        for msg in msgs:
            queue.enqueue(msg)

        # Verify type identification
        msg1 = queue.dequeue()
        assert isinstance(msg1, Send)
        assert msg1.pair_name == "t1"

        msg2 = queue.dequeue()
        assert isinstance(msg2, Receive)
        assert msg2.pair_name == "t1"

        msg3 = queue.dequeue()
        assert isinstance(msg3, Register)
        assert msg3.handler_id == "h1"

    # ==================== Batch Operations ====================

    def test_dequeue_batch(self, queue):
        """Test batch dequeue operation."""
        # Enqueue 10 messages
        for i in range(10):
            queue.enqueue(Send(pair_name=f"t{i}", timestamp=time.time()))

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
            queue.enqueue(Receive(pair_name=f"t{i}", timestamp=1.0))

        # Request 10 but only 3 available
        batch = queue.dequeue_batch(max_count=10)
        assert len(batch) == 3
        assert queue.is_empty()

    def test_peek(self, queue):
        """Test peek does not modify queue."""
        msg = Receive(pair_name="t1", timestamp=1.0)
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
            queue.enqueue(Receive(pair_name=f"t{i}", timestamp=1.0))

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
            queue.enqueue(Send(pair_name=f"t{i}", timestamp=time.time()))

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
        q1.enqueue(Send(pair_name="t1", timestamp=1.0))
        q1.enqueue(Receive(pair_name="t2", timestamp=2.0))
        assert q1.size() == 2
        q1.close()

        # Second queue instance (reopen same file)
        q2 = CommandQueue(temp_lmdb_path)
        assert q2.size() == 2

        msg1 = q2.dequeue()
        assert isinstance(msg1, Send)
        assert msg1.pair_name == "t1"

        msg2 = q2.dequeue()
        assert isinstance(msg2, Receive)
        assert msg2.pair_name == "t2"

        assert q2.is_empty()
        q2.close()

    def test_destroy(self, temp_lmdb_path):
        """Test destroy() completely removes LMDB files."""
        # Create queue and add some data
        q = CommandQueue(temp_lmdb_path)
        q.enqueue(Send(pair_name="t1", timestamp=1.0))
        q.enqueue(Receive(pair_name="t2", timestamp=2.0))
        assert q.size() == 2

        # Verify files exist
        assert os.path.exists(temp_lmdb_path), "Main DB file should exist"
        lock_file = temp_lmdb_path + "-lock"
        assert os.path.exists(lock_file), "Lock file should exist"

        # Destroy queue
        q.destroy()

        # Verify environment is closed
        assert q.env is None, "Environment should be None after destroy"

        # Verify files are deleted
        assert not os.path.exists(temp_lmdb_path), "Main DB file should be deleted"
        assert not os.path.exists(lock_file), "Lock file should be deleted"

    def test_destroy_idempotent(self, temp_lmdb_path):
        """Test destroy() can be called multiple times safely."""
        q = CommandQueue(temp_lmdb_path)
        q.enqueue(Send(pair_name="t1", timestamp=1.0))

        # First destroy
        q.destroy()
        assert not os.path.exists(temp_lmdb_path)

        # Second destroy (should not raise error)
        q.destroy()  # Files already gone, should be silent
