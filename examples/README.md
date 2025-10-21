# Tensor Bus Examples

This directory contains example implementations demonstrating the use of Tensor Bus middleware for efficient tensor transmission in reinforcement learning systems.

## Overview

Tensor Bus is a framework-agnostic middleware for transferring tensor data between training and inference components. It provides synchronization primitives and supports both point-to-point and distributed tensor transfers.

## Examples

### 1. Training Example (`train_example.py`)

Demonstrates how a training loop integrates with the Tensor Bus middleware:

- Synchronizes with middleware state to ensure safe weight updates
- Signals when new weights are available for inference
- Uses RPC client to communicate with middleware service

**Key Features:**
- Forward/backward pass execution
- Optimizer step coordination
- Middleware synchronization

### 2. Inference Example (`inference_example.py`)

Shows how an inference engine coordinates with the middleware:

- Pauses inference when receiving new weights
- Signals readiness to the middleware
- Resumes inference after weight update

**Key Features:**
- Event-driven architecture
- Inference pause/resume mechanism
- Middleware state synchronization

### 3. Middleware Example (`middleware_example.py`)

Illustrates the middleware coordination layer:

- Handles background tensor transfers
- Manages state across multiple inference servers
- Coordinates point-to-point transfers between nodes

**Key Features:**
- Background transfer threads
- Multi-server coordination
- Event-based command handling

## Basic Usage

### Registering a Tensor Pair

```python
from etha import TensorBus

# Initialize Tensor Bus
bus = TensorBus()

# Register a tensor pair for transfer
import torch
tensor = torch.randn(100, 100)
handler = bus.register_pair("weights", tensor)

# Writer side: publish data
handler.put()

# Reader side: get data
with handler.get() as remote_tensor:
    # Use the tensor
    model.load_state_dict(remote_tensor)
# Lock automatically released
```

### Batch Operations

```python
# Register multiple pairs
handler1 = bus.register_pair("layer1", tensor1)
handler2 = bus.register_pair("layer2", tensor2)
handler3 = bus.register_pair("layer3", tensor3)

# Create batched handler
batched = bus.create_batched_handler([handler1, handler2, handler3])

# Put all at once
batched.put()

# Get all at once
with batched.get() as tensors:
    for i, tensor in enumerate(tensors):
        print(f"Layer {i+1}: {tensor.shape}")
```

### Distributed Transfer

```python
from torch.distributed._tensor import DeviceMesh, Shard

# Create device mesh and placement
mesh = DeviceMesh("cuda", torch.arange(4).reshape(2, 2))
placement = Shard(0)

# Register with distributed parameters
handler = bus.register_pair(
    "distributed_weights",
    tensor,
    mesh=mesh,
    placement=placement
)
```

## Design Principles

1. **Low Latency**: Minimize transfer overhead through zero-copy mechanisms
2. **Framework Agnostic**: Support multiple deep learning frameworks
3. **Synchronization Primitives**: Provide RW locks without managing tensor memory
4. **Peer-to-Peer**: Enable direct transfers between components

## API Reference

### PairHandler Methods

- `put()`: Publish data (writer side)
- `get()`: Get data (reader side) - returns context manager
- `is_ready(write=True, timeout_ms=-1)`: Check readiness
- `wait_ready(write=False, timeout_ms=-1)`: Wait until ready

### BatchedPairHandler Methods

- `put()`: Publish all pairs
- `get()`: Get all pairs - returns context manager with list of tensors
- `is_ready(write=True, timeout_ms=-1)`: Check if all pairs are ready
- `wait_ready(write=False, timeout_ms=-1)`: Wait until all pairs are ready

## Notes

- The example RPC and InferServer implementations are placeholders
- In production, implement actual communication mechanisms (HTTP, gRPC, ZMQ, etc.)
- Consider using LMDB + CUDA IPC for zero-copy transfers as suggested in the RFC
- Event mechanisms should be implemented using appropriate queuing systems
