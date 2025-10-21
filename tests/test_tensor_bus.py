"""Tests for Tensor Bus components."""

import time
import threading

import torch
import pytest

from etha import TensorBus, PairHandler, BatchedPairHandler


def test_pair_handler_initialization():
    """Test PairHandler initialization."""
    tensor = torch.randn(10, 10)
    handler = PairHandler("test_pair", tensor)
    
    assert handler.pair_name == "test_pair"
    assert handler.tensor is tensor
    assert handler.mesh is None
    assert handler.placement is None


def test_pair_handler_put_get():
    """Test basic put/get operations."""
    tensor = torch.randn(10, 10)
    handler = PairHandler("test_pair", tensor)
    
    # Put data
    handler.put()
    
    # Get data
    with handler.get() as remote_tensor:
        assert torch.equal(remote_tensor, tensor)


def test_pair_handler_concurrent_reads():
    """Test multiple concurrent readers."""
    tensor = torch.randn(10, 10)
    handler = PairHandler("test_pair", tensor)
    handler.put()
    
    results = []
    
    def reader():
        with handler.get() as remote_tensor:
            results.append(torch.equal(remote_tensor, tensor))
            time.sleep(0.1)  # Hold the lock briefly
    
    # Start multiple readers
    threads = [threading.Thread(target=reader) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    # All readers should succeed
    assert all(results)
    assert len(results) == 3


def test_pair_handler_write_blocks_read():
    """Test that write operation blocks readers."""
    tensor = torch.randn(10, 10)
    handler = PairHandler("test_pair", tensor)
    
    write_acquired = threading.Event()
    write_released = threading.Event()
    read_attempted = threading.Event()
    read_succeeded = threading.Event()
    
    def writer():
        handler._write_lock.acquire()
        write_acquired.set()
        time.sleep(0.2)  # Hold write lock
        handler._write_lock.release()
        write_released.set()
    
    def reader():
        read_attempted.set()
        with handler.get() as _:
            read_succeeded.set()
    
    # Start writer first
    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    
    # Wait for writer to acquire lock
    write_acquired.wait()
    
    # Start reader
    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    
    # Wait briefly for reader to attempt
    time.sleep(0.1)
    read_attempted.wait()
    
    # Reader should not have succeeded yet
    assert not read_succeeded.is_set()
    
    # Wait for write to release
    write_released.wait()
    reader_thread.join()
    
    # Now reader should have succeeded
    assert read_succeeded.is_set()
    writer_thread.join()


def test_pair_handler_is_ready():
    """Test is_ready method."""
    tensor = torch.randn(10, 10)
    handler = PairHandler("test_pair", tensor)
    
    # Should be ready for write initially
    assert handler.is_ready(write=True)
    
    # After put, should be ready for read
    handler.put()
    assert handler.is_ready(write=False)


def test_pair_handler_wait_ready():
    """Test wait_ready method."""
    tensor = torch.randn(10, 10)
    handler = PairHandler("test_pair", tensor)
    
    # Should be ready for write
    assert handler.wait_ready(write=True, timeout_ms=100)
    
    # Put data
    handler.put()
    
    # Should be ready for read
    assert handler.wait_ready(write=False, timeout_ms=100)


def test_batched_pair_handler():
    """Test BatchedPairHandler operations."""
    tensor1 = torch.randn(10, 10)
    tensor2 = torch.randn(20, 20)
    tensor3 = torch.randn(5, 5)
    
    handler1 = PairHandler("pair1", tensor1)
    handler2 = PairHandler("pair2", tensor2)
    handler3 = PairHandler("pair3", tensor3)
    
    batched = BatchedPairHandler([handler1, handler2, handler3])
    
    # Put all
    batched.put()
    
    # Get all
    with batched.get() as tensors:
        assert len(tensors) == 3
        assert torch.equal(tensors[0], tensor1)
        assert torch.equal(tensors[1], tensor2)
        assert torch.equal(tensors[2], tensor3)


def test_batched_pair_handler_is_ready():
    """Test BatchedPairHandler is_ready."""
    tensor1 = torch.randn(10, 10)
    tensor2 = torch.randn(20, 20)
    
    handler1 = PairHandler("pair1", tensor1)
    handler2 = PairHandler("pair2", tensor2)
    
    batched = BatchedPairHandler([handler1, handler2])
    
    # Should be ready for write
    assert batched.is_ready(write=True)
    
    # Put all
    batched.put()
    
    # Should be ready for read
    assert batched.is_ready(write=False)


def test_batched_pair_handler_wait_ready():
    """Test BatchedPairHandler wait_ready."""
    tensor1 = torch.randn(10, 10)
    tensor2 = torch.randn(20, 20)
    
    handler1 = PairHandler("pair1", tensor1)
    handler2 = PairHandler("pair2", tensor2)
    
    batched = BatchedPairHandler([handler1, handler2])
    
    # Should be ready for write
    assert batched.wait_ready(write=True, timeout_ms=100)
    
    # Put all
    batched.put()
    
    # Should be ready for read
    assert batched.wait_ready(write=False, timeout_ms=100)


def test_tensor_bus_register_pair():
    """Test TensorBus register_pair."""
    bus = TensorBus()
    tensor = torch.randn(10, 10)
    
    handler = bus.register_pair("test_pair", tensor)
    
    assert isinstance(handler, PairHandler)
    assert handler.pair_name == "test_pair"
    assert handler.tensor is tensor


def test_tensor_bus_register_same_pair_twice():
    """Test registering the same pair name twice returns same handler."""
    bus = TensorBus()
    tensor1 = torch.randn(10, 10)
    tensor2 = torch.randn(20, 20)
    
    handler1 = bus.register_pair("test_pair", tensor1)
    handler2 = bus.register_pair("test_pair", tensor2)
    
    # Should return the same handler
    assert handler1 is handler2


def test_tensor_bus_create_batched_handler():
    """Test TensorBus create_batched_handler."""
    bus = TensorBus()
    
    tensor1 = torch.randn(10, 10)
    tensor2 = torch.randn(20, 20)
    
    handler1 = bus.register_pair("pair1", tensor1)
    handler2 = bus.register_pair("pair2", tensor2)
    
    batched = bus.create_batched_handler([handler1, handler2])
    
    assert isinstance(batched, BatchedPairHandler)
    assert len(batched.pair_handlers) == 2


def test_tensor_bus_with_distributed_params():
    """Test TensorBus with DeviceMesh and Placement parameters."""
    from torch.distributed._tensor import Shard

    bus = TensorBus()
    tensor = torch.randn(10, 10)

    # Test with None values (point-to-point mode)
    handler = bus.register_pair("dist_pair", tensor, mesh=None, placement=None)

    assert handler.mesh is None
    assert handler.placement is None

    # Test with placement parameter (even without actual mesh)
    placement = Shard(0)
    handler2 = bus.register_pair("dist_pair2", tensor, mesh=None, placement=placement)

    assert handler2.mesh is None
    assert handler2.placement is placement


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
