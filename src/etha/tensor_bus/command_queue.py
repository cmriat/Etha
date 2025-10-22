"""LMDB-based Command Queue for Tensor Bus."""

import struct

import lmdb
import msgspec
from upath import UPath

from .messages import Message


# TODO: really, could be improved, 1. limited buffer, maybe circular queue 2. single writer multiple reader optimization
class CommandQueue:
    """LMDB based command queue.

    Features:
    - Monotonic increasing ID as key (FIFO guarantee)
    - LMDB transaction ensures atomicity
    - Simple, reliable, no extra synchronization needed
    - Supports all Message types (via tagged union)
    """

    # Class-level encoder/decoder for performance (reuse instances)
    _encoder = msgspec.msgpack.Encoder()
    _decoder = msgspec.msgpack.Decoder(Message)

    def __init__(self, lmdb_path: str = "/tmp/tensor_bus.lmdb"):
        self.lmdb_path = UPath(lmdb_path)
        self.env = lmdb.open(
            str(self.lmdb_path),
            max_dbs=2,
            map_size=1 << 30,  # 1GB
            subdir=False,
            lock=True,
        )

        # Command queue database
        self.queue_db = self.env.open_db(b"command_queue")

        # Initialize head/tail pointers
        self._init_pointers()

    def _init_pointers(self):
        """Initialize queue head/tail pointers."""
        with self.env.begin(write=True, db=self.queue_db) as txn:
            if txn.get(b"__head__") is None:
                txn.put(b"__head__", struct.pack("Q", 0))  # Read pointer
                txn.put(b"__tail__", struct.pack("Q", 0))  # Write pointer

    def enqueue(self, msg: Message) -> int:
        """Enqueue a message.

        Args:
            msg: Any Message type (PutCommand, GetCommand, etc.)

        Returns:
            Message ID (queue position)
        """
        with self.env.begin(write=True, db=self.queue_db) as txn:
            # Get current tail
            tail_bytes = txn.get(b"__tail__")
            tail = struct.unpack("Q", tail_bytes)[0]

            # Serialize and write message (key = 8-byte tail representation)
            key = struct.pack("Q", tail)
            txn.put(key, self._encoder.encode(msg))

            # Increment tail
            txn.put(b"__tail__", struct.pack("Q", tail + 1))

            return tail

    def dequeue(self) -> Message | None:
        """Dequeue a message.

        Returns:
            Message object (auto-detected type), or None if queue is empty
        """
        with self.env.begin(write=True, db=self.queue_db) as txn:
            # Get head and tail
            head_bytes = txn.get(b"__head__")
            tail_bytes = txn.get(b"__tail__")

            head = struct.unpack("Q", head_bytes)[0]
            tail = struct.unpack("Q", tail_bytes)[0]

            # Check if empty
            if head >= tail:
                return None

            # Read command
            key = struct.pack("Q", head)
            cmd_data = txn.get(key)

            if cmd_data is None:
                # Data corrupted, skip
                txn.put(b"__head__", struct.pack("Q", head + 1))
                return None

            try:
                cmd = self._decoder.decode(cmd_data)
            except msgspec.DecodeError:
                # Data corrupted, skip
                txn.put(b"__head__", struct.pack("Q", head + 1))
                return None

            # Delete command (save space)
            txn.delete(key)

            # Increment head
            txn.put(b"__head__", struct.pack("Q", head + 1))

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
            head_bytes = txn.get(b"__head__")
            tail_bytes = txn.get(b"__tail__")

            head = struct.unpack("Q", head_bytes)[0]
            tail = struct.unpack("Q", tail_bytes)[0]

            # Calculate actual dequeue count
            count = min(max_count, tail - head)

            for i in range(count):
                key = struct.pack("Q", head + i)
                cmd_data = txn.get(key)

                if cmd_data:
                    try:
                        cmd = self._decoder.decode(cmd_data)
                        commands.append(cmd)
                        txn.delete(key)
                    except msgspec.DecodeError:
                        # Skip corrupted data
                        txn.delete(key)
                        continue

            # Update head
            if count > 0:
                txn.put(b"__head__", struct.pack("Q", head + count))

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

            key = struct.pack("Q", head)
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

            # Delete all commands
            cursor = txn.cursor()
            for key, _ in cursor:
                if not key.startswith(b"__"):
                    txn.delete(key)

    def close(self):
        """Close queue."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def destroy(self):
        """Completely destroy the queue and delete all LMDB files.

        WARNING: This is irreversible. The queue instance will be unusable after this call.
        Use this for cleanup in tests or when you want to start fresh.

        This method will:
        1. Close the LMDB environment
        2. Delete the main database file
        3. Delete the lock file

        Errors are silently ignored (best-effort deletion).
        """
        # Close environment first
        self.close()

        # Delete LMDB files (silent on errors)
        try:
            self.lmdb_path.unlink(missing_ok=True)
        except Exception:
            pass  # Best effort

        try:
            lock_file = UPath(str(self.lmdb_path) + "-lock")
            lock_file.unlink(missing_ok=True)
        except Exception:
            pass  # Best effort

    def __repr__(self):
        return f"CommandQueue(path={self.lmdb_path}, size={self.size()})"
