"""Tensor Bus: High-efficiency tensor transmission middleware for RL components."""

from __future__ import annotations

import threading
from typing import List, Iterator, Optional
from contextlib import contextmanager

import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.placement_types import Placement


class PairHandler:
    """Handler for a registered tensor pair enabling synchronized data transfer.
    
    Provides synchronization primitives (lock operations) without managing tensor memory.
    Designed for peer-to-peer tensor communication between training and inference components.
    """

    def __init__(
        self,
        pair_name: str,
        tensor: torch.Tensor,
        mesh: Optional[DeviceMesh] = None,
        placement: Optional[Placement] = None,
    ):
        """Initialize a PairHandler.
        
        Args:
            pair_name: Unique identifier for this pair
            tensor: The tensor to be transferred
            mesh: Optional DeviceMesh for distributed transfer
            placement: Optional Placement for data layout
        """
        self.pair_name = pair_name
        self.tensor = tensor
        self.mesh = mesh
        self.placement = placement
        
        # RW lock implementation
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._reader_count = 0
        self._reader_count_lock = threading.Lock()
        self._ready_event = threading.Event()
        self._ready_event.set()  # Initially ready for write

    def put(self) -> None:
        """Publish data (Writer side).
        
        Automatically manages locks to ensure safe concurrent access.
        """
        with self._write_lock:
            # Mark data as ready for readers
            self._ready_event.set()

    @contextmanager
    def get(self) -> Iterator[torch.Tensor]:
        """Get data (Reader side).
        
        Usage:
            with pair.get() as remote:
                model.load_state_dict(remote)
            # Automatically releases read lock
            
        Yields:
            torch.Tensor: The remote tensor data
        """
        # Acquire read lock
        with self._reader_count_lock:
            self._reader_count += 1
            if self._reader_count == 1:
                self._write_lock.acquire()
        
        try:
            yield self.tensor
        finally:
            # Release read lock
            with self._reader_count_lock:
                self._reader_count -= 1
                if self._reader_count == 0:
                    self._write_lock.release()

    def is_ready(self, write: bool = True, timeout_ms: int = -1) -> bool:
        """Check if the pair is ready for read or write.
        
        Args:
            write: If True, check write readiness; if False, check read readiness
            timeout_ms: Timeout in milliseconds (-1 for no timeout)
            
        Returns:
            bool: True if ready, False otherwise
        """
        timeout_s = None if timeout_ms < 0 else timeout_ms / 1000.0
        
        if write:
            # For write, check if no readers are active
            acquired = self._write_lock.acquire(blocking=False)
            if acquired:
                self._write_lock.release()
                return True
            return False
        else:
            # For read, check if data is ready
            return self._ready_event.wait(timeout=timeout_s)

    def wait_ready(self, write: bool = False, timeout_ms: int = -1) -> bool:
        """Wait until the pair is ready for read or write.
        
        Args:
            write: If True, wait for write readiness; if False, wait for read readiness
            timeout_ms: Timeout in milliseconds (-1 for no timeout)
            
        Returns:
            bool: True if became ready, False if timed out
        """
        timeout_s = None if timeout_ms < 0 else timeout_ms / 1000.0
        
        if write:
            # For write, wait until we can acquire the write lock
            acquired = self._write_lock.acquire(blocking=True, timeout=timeout_s or -1)
            if acquired:
                self._write_lock.release()
                return True
            return False
        else:
            # For read, wait until data is ready
            return self._ready_event.wait(timeout=timeout_s)


class BatchedPairHandler:
    """Batched handler for multiple PairHandlers to enable batch operations."""

    def __init__(self, pair_handlers: List[PairHandler]):
        """Initialize a BatchedPairHandler.
        
        Args:
            pair_handlers: List of PairHandlers to batch together
        """
        self.pair_handlers = pair_handlers

    def put(self) -> None:
        """Publish data for all pairs in the batch."""
        for handler in self.pair_handlers:
            handler.put()

    @contextmanager
    def get(self) -> Iterator[List[torch.Tensor]]:
        """Get data from all pairs in the batch.
        
        Usage:
            with batched_pair.get() as remote_tensors:
                for i, tensor in enumerate(remote_tensors):
                    # Process each tensor
                    pass
            # Automatically releases all read locks
            
        Yields:
            List[torch.Tensor]: List of remote tensor data
        """
        # Acquire all read locks
        contexts = [handler.get() for handler in self.pair_handlers]
        tensors = []
        
        # Enter all contexts
        for ctx in contexts:
            tensors.append(ctx.__enter__())
        
        try:
            yield tensors
        finally:
            # Exit all contexts in reverse order
            for ctx in reversed(contexts):
                ctx.__exit__(None, None, None)

    def is_ready(self, write: bool = True, timeout_ms: int = -1) -> bool:
        """Check if all pairs in the batch are ready.
        
        Args:
            write: If True, check write readiness; if False, check read readiness
            timeout_ms: Timeout in milliseconds (-1 for no timeout)
            
        Returns:
            bool: True if all pairs are ready, False otherwise
        """
        return all(handler.is_ready(write, timeout_ms) for handler in self.pair_handlers)

    def wait_ready(self, write: bool = False, timeout_ms: int = -1) -> bool:
        """Wait until all pairs in the batch are ready.
        
        Args:
            write: If True, wait for write readiness; if False, wait for read readiness
            timeout_ms: Timeout in milliseconds (-1 for no timeout)
            
        Returns:
            bool: True if all became ready, False if any timed out
        """
        return all(handler.wait_ready(write, timeout_ms) for handler in self.pair_handlers)


class TensorBus:
    """Tensor Bus middleware for efficient tensor transmission between components.
    
    Provides a framework-agnostic API for transferring tensor data between
    training and inference components in reinforcement learning systems.
    """

    def __init__(self):
        """Initialize the TensorBus."""
        self._pairs = {}
        self._lock = threading.Lock()

    def register_pair(
        self,
        pair_name: str,
        tensor: torch.Tensor,
        mesh: Optional[DeviceMesh] = None,
        placement: Optional[Placement] = None,
    ) -> PairHandler:
        """Register a tensor pair for peer-to-peer transfer.
        
        Args:
            pair_name: Unique identifier for this pair
            tensor: The tensor to be transferred
            mesh: Optional DeviceMesh for distributed transfer
                  If None, degrades to point-to-point transfer
            placement: Optional Placement for data layout
                       If None, degrades to point-to-point transfer
                       
        Returns:
            PairHandler: Handler for the registered pair
            
        Notes:
            - If both mesh and placement are None: point-to-point transfer
            - If both mesh and placement are provided: distributed transfer
            - If one side is None: broadcast
        """
        with self._lock:
            if pair_name in self._pairs:
                return self._pairs[pair_name]
            
            handler = PairHandler(pair_name, tensor, mesh, placement)
            self._pairs[pair_name] = handler
            return handler

    def create_batched_handler(self, pair_handlers: List[PairHandler]) -> BatchedPairHandler:
        """Create a batched handler for multiple pairs.
        
        Args:
            pair_handlers: List of PairHandlers to batch together
            
        Returns:
            BatchedPairHandler: Batched handler for batch operations
        """
        return BatchedPairHandler(pair_handlers)
