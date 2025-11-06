"""LMDB-based Command Queue for Tensor Bus."""

import struct
import logging

import lmdb
import msgspec
import posix_ipc
from upath import UPath

from .commands import Message

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    """Raised when trying to enqueue to a full CommandQueue with block=False."""

    pass


# TODO: single writer multiple reader optimization
class CommandQueue:
    """LMDB based command queue."""

    # Class-level encoder/decoder for performance (reuse instances)
    _encoder = msgspec.msgpack.Encoder()
    _decoder = msgspec.msgpack.Decoder(Message)

    def __init__(self, lmdb_path: str = "/tmp/tensor_bus.lmdb", capacity: int = 50000):
        self.lmdb_path = UPath(lmdb_path)
        self.capacity = capacity  # Fixed circular queue capacity
        self.env = lmdb.open(
            str(self.lmdb_path),
            max_dbs=2,
            map_size=1 << 30,
            subdir=False,
            lock=True,
        )

        # Command queue database
        self.queue_db = self.env.open_db(b"command_queue")
        self._init_pointers()

        # POSIX semaphores:
        # _sem: notify consumers that data is available (count = number of items)
        # _space_sem: notify producers that space is available (count = number of free slots)
        # _ready_sem: synchronization barrier for initialization (leader election)
        self.sem_name = f"/cq_{self.lmdb_path.stem}"
        self.space_sem_name = f"/cq_space_{self.lmdb_path.stem}"
        self.ready_name = f"/cq_ready_{self.lmdb_path.stem}"

        try:
            # Try to become leader (create ready semaphore)
            ready = posix_ipc.Semaphore(self.ready_name, flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL, initial_value=0)

            # Leader: create both resource semaphores atomically
            try:
                current_size = self.size()
                self._sem = posix_ipc.Semaphore(
                    self.sem_name, flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL, initial_value=current_size
                )
                self._space_sem = posix_ipc.Semaphore(
                    self.space_sem_name,
                    flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL,
                    initial_value=capacity - current_size,
                )
                # Signal that resources are ready
                ready.release()
                ready.close()
            except Exception:
                # Leader failed during creation, cleanup everything
                try:
                    posix_ipc.unlink_semaphore(self.sem_name)
                    posix_ipc.unlink_semaphore(self.space_sem_name)
                    ready.unlink()
                except Exception:
                    pass
                raise
        except posix_ipc.ExistentialError:
            # Follower: wait for leader to finish
            ready = posix_ipc.Semaphore(self.ready_name)
            ready.acquire()
            ready.release()  # Release for other followers
            ready.close()

            self._sem = posix_ipc.Semaphore(self.sem_name)
            self._space_sem = posix_ipc.Semaphore(self.space_sem_name)

    def _init_pointers(self):
        """Initialize queue head/tail pointers."""
        with self.env.begin(write=True, db=self.queue_db) as txn:
            if txn.get(b"__head__") is None:
                txn.put(b"__head__", struct.pack("Q", 0))  # Read pointer
                txn.put(b"__tail__", struct.pack("Q", 0))  # Write pointer

    def enqueue(self, msg: Message, *, block: bool = True, timeout: float | None = None) -> int:
        """Enqueue a message.

        Args:
            msg: Any Message type (PutCommand, GetCommand, etc.)
            block: If True, block when queue is full. If False, raise QueueFullError.
            timeout: Timeout in seconds for blocking wait (None = infinite)

        Returns:
            Message ID (queue position)

        Raises:
            QueueFullError: If queue is full and block=False, or timeout expires
        """
        # Wait for space to become available
        if block:
            # Blocking wait
            try:
                self._space_sem.acquire(timeout=timeout)
            except posix_ipc.BusyError:
                raise QueueFullError(f"Queue is full (capacity={self.capacity})") from None
        else:
            # Non-blocking: just try to acquire immediately
            try:
                self._space_sem.acquire(timeout=0)
            except posix_ipc.BusyError:
                raise QueueFullError(f"Queue is full (capacity={self.capacity})") from None

        with self.env.begin(write=True, db=self.queue_db) as txn:
            # Get current tail
            tail_bytes = txn.get(b"__tail__")
            tail = struct.unpack("Q", tail_bytes)[0]

            # Circular indexing: key is always in [0, capacity)
            idx = tail % self.capacity
            key = struct.pack("Q", idx)
            txn.put(key, self._encoder.encode(msg))

            # Increment tail
            txn.put(b"__tail__", struct.pack("Q", tail + 1))

            self._sem.release()  # Notify consumers that data is available
            return tail

    def dequeue(self, *, block: bool = False, timeout: float | None = None) -> Message | None:
        """Dequeue a message.

        Returns:
            Message object (auto-detected type), or None if queue is empty
        """
        try:
            if block:
                self._sem.acquire(timeout=timeout)
        except posix_ipc.BusyError:
            return None

        return self._try_dequeue_once()

    def dequeue_batch(self, max_count: int = 32) -> list[Message]:
        """Batch dequeue (improve throughput).

        Args:
            max_count: Maximum number of messages to dequeue

        Returns:
            List of messages
        """
        commands = []

        with self.env.begin(write=True, db=self.queue_db) as txn:
            head_bytes = txn.get(b"__head__")
            tail_bytes = txn.get(b"__tail__")

            head = struct.unpack("Q", head_bytes)[0]
            tail = struct.unpack("Q", tail_bytes)[0]

            # Calculate actual dequeue count
            count = min(max_count, tail - head)

            for i in range(count):
                # Circular indexing: key is always in [0, capacity)
                idx = (head + i) % self.capacity
                key = struct.pack("Q", idx)
                cmd_data = txn.get(key)

                if cmd_data:
                    try:
                        cmd = self._decoder.decode(cmd_data)
                        commands.append(cmd)
                        # NO DELETE! Just increment head, let next enqueue overwrite
                    except msgspec.DecodeError:
                        # Skip corrupted data - still counts as freeing space
                        continue

            # Update head
            if count > 0:
                txn.put(b"__head__", struct.pack("Q", head + count))

        # Notify producers that space is now available for each dequeued item
        for _ in range(count):
            self._space_sem.release()

        return commands

    def size(self) -> int:
        """Return queue length."""
        with self.env.begin(db=self.queue_db) as txn:
            head = struct.unpack("Q", txn.get(b"__head__"))[0]
            tail = struct.unpack("Q", txn.get(b"__tail__"))[0]
            return tail - head

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0

    def peek(self) -> Message | None:
        """View front message (without dequeuing)."""
        with self.env.begin(db=self.queue_db) as txn:
            head_bytes = txn.get(b"__head__")
            tail_bytes = txn.get(b"__tail__")

            head = struct.unpack("Q", head_bytes)[0]
            tail = struct.unpack("Q", tail_bytes)[0]

            if head >= tail:
                return None

            # Circular indexing: key is always in [0, capacity)
            idx = head % self.capacity
            key = struct.pack("Q", idx)
            cmd_data = txn.get(key)

            if cmd_data:
                try:
                    return self._decoder.decode(cmd_data)
                except msgspec.DecodeError:
                    return None
            return None

    def clear(self):
        """Clear queue (for testing)."""
        with self.env.begin(write=True, db=self.queue_db) as txn:
            # Reset pointers
            txn.put(b"__head__", struct.pack("Q", 0))
            txn.put(b"__tail__", struct.pack("Q", 0))

            # Delete all commands (optional but cleaner)
            cursor = txn.cursor()
            for key, _ in cursor:
                if not key.startswith(b"__"):
                    txn.delete(key)

        # Drain semaphore count to 0
        while True:
            try:
                self._sem.acquire(timeout=0)
            except posix_ipc.BusyError:
                break

        # Reset space semaphore to full capacity
        while True:
            try:
                self._space_sem.acquire(timeout=0)
            except posix_ipc.BusyError:
                break
        # Re-fill space semaphore to full capacity
        for _ in range(self.capacity):
            self._space_sem.release()

    def _try_dequeue_once(self) -> Message | None:
        with self.env.begin(write=True, db=self.queue_db) as txn:
            head_bytes = txn.get(b"__head__")
            tail_bytes = txn.get(b"__tail__")

            head = struct.unpack("Q", head_bytes)[0]
            tail = struct.unpack("Q", tail_bytes)[0]

            if head >= tail:
                return None

            # Circular indexing: key is always in [0, capacity)
            idx = head % self.capacity
            key = struct.pack("Q", idx)
            cmd_data = txn.get(key)
            if cmd_data is None:
                txn.put(b"__head__", struct.pack("Q", head + 1))
                return None

            try:
                cmd = self._decoder.decode(cmd_data)
            except msgspec.DecodeError:
                txn.put(b"__head__", struct.pack("Q", head + 1))
                return None

            txn.put(b"__head__", struct.pack("Q", head + 1))

            self._space_sem.release()
            return cmd

    def close(self, destroy: bool = True):
        """Close queue connection.

        Args:
            destroy: If True (default), completely remove all resources (LMDB files, semaphores).
                     If False, only close handles in this process (safe for multi-process).

        Default behavior (destroy=True):
            - Most common case: single process or final cleanup
            - Unlinks semaphores from the system
            - Deletes LMDB database files
            - Ensures no resource leaks

        Multi-process mode (destroy=False):
            - Use when other processes are still using the same queue
            - Only closes handles in this process
            - Resources remain available for other processes
            - Last process should call close() or close(destroy=True) to clean up

        Examples:
            # Single process (default)
            q = CommandQueue('/tmp/test.lmdb')
            q.enqueue(msg)
            q.close()  # Automatically cleans up everything

            # Multi-process
            # Worker process
            q = CommandQueue('/tmp/bus.lmdb')
            q.enqueue(msg)
            q.close(destroy=False)  # Don't delete, other processes need it

            # Main process (last one)
            q = CommandQueue('/tmp/bus.lmdb')
            q.close()  # Final cleanup, removes all resources
        """
        # If destroy=True, unlink semaphores first (before closing handles)
        if destroy:
            if self._sem is not None:
                try:
                    self._sem.unlink()
                except posix_ipc.ExistentialError:
                    pass

            if self._space_sem is not None:
                try:
                    self._space_sem.unlink()
                except posix_ipc.ExistentialError:
                    pass

            try:
                posix_ipc.unlink_semaphore(self.ready_name)
            except posix_ipc.ExistentialError:
                pass

        # Close semaphore handles (safe to call multiple times)
        if self._sem is not None:
            try:
                self._sem.close()
            except posix_ipc.ExistentialError:
                pass
            self._sem = None

        if self._space_sem is not None:
            try:
                self._space_sem.close()
            except posix_ipc.ExistentialError:
                pass
            self._space_sem = None

        # Close LMDB environment
        if self.env is not None:
            self.env.close()
            self.env = None

        # If destroy=True, delete LMDB files
        if destroy:
            try:
                self.lmdb_path.unlink()
            except Exception:
                pass

            try:
                lock_file = UPath(str(self.lmdb_path) + "-lock")
                lock_file.unlink()
            except Exception:
                pass

    def __repr__(self):
        return f"CommandQueue(path={self.lmdb_path}, size={self.size()})"
