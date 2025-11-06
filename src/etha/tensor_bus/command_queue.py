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

        # Store semaphore names
        self.sem_name = f"/cq_{self.lmdb_path.stem}"  # Semaphore for available items
        self.space_sem_name = f"/cq_space_{self.lmdb_path.stem}"  # Semaphore for free space
        self.ready_name = f"/cq_ready_{self.lmdb_path.stem}"  # Semaphore for initialization readiness (leader election)

        try:
            # Try to become leader (create ready semaphore)
            ready = posix_ipc.Semaphore(self.ready_name, flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL, initial_value=0)

            # Leader: open LMDB and initialize if needed
            self._init_lmdb_environment()

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

            # Follower: open LMDB (should already be initialized)
            self._init_lmdb_environment()

            self._sem = posix_ipc.Semaphore(self.sem_name)
            self._space_sem = posix_ipc.Semaphore(self.space_sem_name)

    def _init_lmdb_environment(self):
        """Initialize LMDB environment and queue database."""
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

    def _init_pointers(self):
        """Initialize queue head/tail pointers."""
        with self.env.begin(write=True, db=self.queue_db) as txn:
            if txn.get(b"__head__") is None:
                self._set_head(txn, 0)
                self._set_tail(txn, 0)

    def _get_head(self, txn) -> int:
        """Helper: get head pointer from transaction."""
        return struct.unpack("Q", txn.get(b"__head__"))[0]

    def _get_tail(self, txn) -> int:
        """Helper: get tail pointer from transaction."""
        return struct.unpack("Q", txn.get(b"__tail__"))[0]

    def _set_head(self, txn, value: int):
        """Helper: set head pointer."""
        txn.put(b"__head__", struct.pack("Q", value))

    def _set_tail(self, txn, value: int):
        """Helper: set tail pointer."""
        txn.put(b"__tail__", struct.pack("Q", value))

    def _increment_head(self, txn):
        """Helper: increment head pointer (dequeue operation)."""
        head = self._get_head(txn)
        self._set_head(txn, head + 1)

    def _increment_tail(self, txn):
        """Helper: increment tail pointer (enqueue operation)."""
        tail = self._get_tail(txn)
        self._set_tail(txn, tail + 1)

    def _circular_key(self, pos: int):
        """Helper: calculate circular queue key for position."""
        idx = pos % self.capacity
        return struct.pack("Q", idx)

    def _decode_message(self, data: bytes):
        """Helper: decode message data, return None if corrupted."""
        try:
            return self._decoder.decode(data)
        except msgspec.DecodeError:
            return None

    def _put_message(self, txn, pos: int, msg: Message):
        """Helper: write message to logical position."""
        key = self._circular_key(pos)
        txn.put(key, self._encoder.encode(msg))

    def _get_message(self, txn, pos: int) -> Message | None:
        """Helper: read message from logical position, return None if not found or corrupted."""
        key = self._circular_key(pos)
        cmd_data = txn.get(key)
        if cmd_data is None:
            return None
        return self._decode_message(cmd_data)

    def _size(self, txn) -> int:
        """Return queue length."""
        return self._get_tail(txn) - self._get_head(txn)

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
            # Get current tail and write message
            tail = self._get_tail(txn)
            self._put_message(txn, tail, msg)

            # Increment tail pointer
            self._increment_tail(txn)

            # Notify consumers that data is available
            self._sem.release()
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

        with self.env.begin(write=True, db=self.queue_db) as txn:
            head = self._get_head(txn)
            if self._size(txn) <= 0:
                return None

            cmd = self._get_message(txn, head)

            self._increment_head(txn)

            if cmd is not None:
                self._space_sem.release()

            return cmd

    def dequeue_batch(self, max_count: int = 32) -> list[Message]:
        """Batch dequeue (improve throughput).

        Args:
            max_count: Maximum number of messages to dequeue

        Returns:
            List of messages
        """
        commands = []

        with self.env.begin(write=True, db=self.queue_db) as txn:
            head = self._get_head(txn)
            tail = self._get_tail(txn)

            count = min(max_count, tail - head)

            for i in range(count):
                # Get message data
                cmd = self._get_message(txn, head + i)
                if cmd:
                    commands.append(cmd)

            # Update head pointer (batch update)
            if count > 0:
                self._set_head(txn, head + count)

        # Notify producers that space is now available for each dequeued item
        for _ in range(count):
            self._space_sem.release()

        return commands

    def size(self) -> int:
        """Return queue length."""
        with self.env.begin(db=self.queue_db) as txn:
            return self._size(txn)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0

    def peek(self) -> Message | None:
        """View front message (without dequeuing)."""
        with self.env.begin(db=self.queue_db) as txn:
            head = self._get_head(txn)
            if self._size(txn) <= 0:
                return None

            # Get message data without dequeuing
            return self._get_message(txn, head)

    def clear(self, delete_commands: bool = True):
        """Clear queue."""
        with self.env.begin(write=True, db=self.queue_db) as txn:
            # Reset pointers
            self._set_head(txn, 0)
            self._set_tail(txn, 0)

            # Delete all commands
            if delete_commands:
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
