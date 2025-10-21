# Tensor Bus Design Documentation

## Overview

Tensor Bus is a middleware for high-efficiency tensor transmission between different computing units in reinforcement learning systems. It addresses the core challenge of enabling efficient tensor transfer between components in complex multi-component RL scenarios.

## Problem Statement

In reinforcement learning, we typically have two types of components: training and inference. Due to the divergence in their workload characteristics, their framework implementations are increasingly specialized. Framework-level interconnection would require significant adaptation work and unnecessary complexity. Therefore, we need a framework-agnostic solution for transferring tensor data between components.

## Design Principles

1. **Low Latency**: Minimize transfer delay to meet real-time requirements in RL
2. **Zero-Copy**: Use zero-copy mechanisms for host-TensorBus communication
3. **Framework Agnostic**: Support multiple deep learning frameworks to avoid vendor lock-in

## Architecture

### Core Components

#### 1. TensorBus (Main Interface)
The central registry and coordinator for tensor pairs.

```python
bus = TensorBus()
handler = bus.register_pair(pair_name, tensor, mesh, placement)
```

#### 2. PairHandler (Synchronization Primitive)
Manages synchronization for a single tensor pair using RW locks.

**Key Methods:**
- `put()`: Writer side - publishes data
- `get()`: Reader side - context manager for safe access
- `is_ready()`: Non-blocking readiness check
- `wait_ready()`: Blocking wait for readiness

**Lock Implementation:**
- Multiple concurrent readers allowed
- Writers get exclusive access
- Automatic lock management via context managers

#### 3. BatchedPairHandler
Enables batch operations on multiple tensor pairs.

```python
batched = bus.create_batched_handler([handler1, handler2, handler3])
batched.put()  # Publish all
with batched.get() as tensors:  # Get all
    # Process tensors
    pass
```

### Data Model

The tensor transfer is modeled as moving a distributed tensor (DTensor) `T` from:
- Source mesh `M_src` with placement `P_src`
- To target mesh `M_target` with placement `P_target`

This general modeling covers various transfer scenarios:
- **Point-to-Point**: Both mesh and placement are None
- **Distributed Transfer**: Both mesh and placement are provided
- **Broadcast**: One side is None

### Synchronization Model

Uses a Reader-Writer (RW) lock pattern:
- **Write Lock**: Exclusive access for updates
- **Read Lock**: Shared access for multiple readers
- **Ready Event**: Signals data availability

```
Writer:                     Reader:
┌──────────┐               ┌──────────┐
│ Acquire  │               │ Wait for │
│ Write    │               │ Ready    │
│ Lock     │               │          │
└────┬─────┘               └────┬─────┘
     │                          │
     ▼                          ▼
┌──────────┐               ┌──────────┐
│ Update   │               │ Acquire  │
│ Data     │               │ Read     │
│          │               │ Lock     │
└────┬─────┘               └────┬─────┘
     │                          │
     ▼                          ▼
┌──────────┐               ┌──────────┐
│ Signal   │───────────────▶│ Read     │
│ Ready    │               │ Data     │
└────┬─────┘               └────┬─────┘
     │                          │
     ▼                          ▼
┌──────────┐               ┌──────────┐
│ Release  │               │ Release  │
│ Lock     │               │ Lock     │
└──────────┘               └──────────┘
```

## Implementation Status

### ✅ Completed (Phase 1)

1. **Core API**
   - PairHandler with RW lock synchronization
   - BatchedPairHandler for batch operations
   - TensorBus registration system

2. **Utility Classes**
   - State management
   - RPC client placeholder
   - InferServer client placeholder

3. **Examples**
   - Training loop integration
   - Inference engine integration
   - Middleware coordination

4. **Testing**
   - 13 comprehensive tests
   - Thread safety validation
   - Concurrent access testing

### 🚧 Future Work (Phase 2+)

1. **Bottom Layer Loop**
   - LMDB + CUDA IPC for host-TensorBus communication
   - Command queue for async operations
   - Loop executor for instruction processing

2. **Strategy Generation**
   - Mark-and-sweep algorithm for simple cases
   - Collision and multi-hop optimization
   - IR representation for strategy optimization

3. **CPU DRAM Buffer Cache** (Extension 1)
   - Cache for cold tensor data when GPU memory is insufficient
   - Different sample IDs for the same column

4. **Distributed Parameter Server** (Extension 2)
   - Multi-version weight management for async RL
   - Version routing across different source meshes
   - Distributed weight serving

## Usage Examples

### Basic Usage

```python
from etha import TensorBus
import torch

# Initialize
bus = TensorBus()
tensor = torch.randn(100, 100)

# Register pair
handler = bus.register_pair("weights", tensor)

# Writer side
handler.put()

# Reader side
with handler.get() as remote:
    model.load_state_dict(remote)
```

### Distributed Transfer

```python
from torch.distributed._tensor import DeviceMesh, Shard

mesh = DeviceMesh("cuda", torch.arange(4).reshape(2, 2))
placement = Shard(0)

handler = bus.register_pair(
    "distributed_weights",
    tensor,
    mesh=mesh,
    placement=placement
)
```

### Batch Operations

```python
handlers = [
    bus.register_pair(f"layer_{i}", tensor)
    for i, tensor in enumerate(tensors)
]

batched = bus.create_batched_handler(handlers)
batched.put()

with batched.get() as remote_tensors:
    for tensor in remote_tensors:
        # Process each tensor
        pass
```

## Integration with Existing Components

The Tensor Bus integrates with the existing communication utilities:

```python
from etha import (
    TensorBus,           # New: High-level API
    get_p2p_map,         # Existing: Strategy generation
    p2p_communicate,     # Existing: P2P transfer
    get_shard_tensor_shape  # Existing: Shape calculation
)
```

The existing `get_p2p_map` and `p2p_communicate` functions can be used within the Tensor Bus middleware for actual data transfer operations.

## Performance Considerations

1. **Zero-Copy**: Future implementation will use CUDA IPC for zero-copy transfers
2. **Lock Overhead**: Minimal - uses Python threading primitives
3. **Context Managers**: Ensure automatic lock release, preventing deadlocks
4. **Batch Operations**: Reduce overhead when transferring multiple tensors

## Testing Strategy

### Unit Tests
- PairHandler synchronization correctness
- BatchedPairHandler batch operations
- TensorBus registration and retrieval

### Integration Tests (Future)
- End-to-end transfer validation
- Distributed mesh communication
- Performance benchmarking

### Concurrency Tests
- Multiple concurrent readers
- Write-read blocking behavior
- Thread safety verification

## API Reference

### TensorBus

```python
class TensorBus:
    def register_pair(
        pair_name: str,
        tensor: torch.Tensor,
        mesh: Optional[DeviceMesh] = None,
        placement: Optional[Placement] = None
    ) -> PairHandler
    
    def create_batched_handler(
        pair_handlers: List[PairHandler]
    ) -> BatchedPairHandler
```

### PairHandler

```python
class PairHandler:
    def put() -> None
    
    @contextmanager
    def get() -> Iterator[torch.Tensor]
    
    def is_ready(write: bool = True, timeout_ms: int = -1) -> bool
    
    def wait_ready(write: bool = False, timeout_ms: int = -1) -> bool
```

### BatchedPairHandler

```python
class BatchedPairHandler:
    def put() -> None
    
    @contextmanager
    def get() -> Iterator[List[torch.Tensor]]
    
    def is_ready(write: bool = True, timeout_ms: int = -1) -> bool
    
    def wait_ready(write: bool = False, timeout_ms: int = -1) -> bool
```

## Conclusion

This implementation provides the foundation for efficient, framework-agnostic tensor transfer in RL systems. The design follows the RFC specification while leaving room for future optimizations like LMDB+CUDA IPC integration and distributed parameter serving.
