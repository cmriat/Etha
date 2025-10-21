# Tensor Bus Quick Start Guide

## Installation

```bash
pip install etha
```

Or for development:
```bash
git clone https://github.com/cmriat/Etha.git
cd Etha
pip install -e .
```

## 5-Minute Tutorial

### 1. Basic Point-to-Point Transfer

The simplest use case - transferring a tensor between two components:

```python
import torch
from etha import TensorBus

# Initialize Tensor Bus
bus = TensorBus()

# Create a tensor
weights = torch.randn(1000, 1000)

# Register the tensor pair
handler = bus.register_pair("model_weights", weights)

# === On the Writer Side (e.g., Training Process) ===
def training_loop():
    for step in range(100):
        # Train model...
        # Update weights...
        
        # Publish updated weights
        handler.put()
        print(f"Step {step}: Weights published")

# === On the Reader Side (e.g., Inference Process) ===
def inference_loop():
    while True:
        # Wait for weights to be ready
        if handler.wait_ready(write=False, timeout_ms=1000):
            # Get the weights safely
            with handler.get() as remote_weights:
                # Use the weights for inference
                model.load_state_dict(remote_weights)
                # Run inference...
```

### 2. Batch Operations

When you need to transfer multiple tensors together:

```python
import torch
from etha import TensorBus

bus = TensorBus()

# Register multiple layers
layer1 = bus.register_pair("layer1", torch.randn(100, 100))
layer2 = bus.register_pair("layer2", torch.randn(200, 200))
layer3 = bus.register_pair("layer3", torch.randn(300, 300))

# Create batch handler
batched = bus.create_batched_handler([layer1, layer2, layer3])

# Writer: publish all layers at once
batched.put()

# Reader: get all layers at once
with batched.get() as layers:
    for i, layer_weights in enumerate(layers):
        print(f"Layer {i+1} shape: {layer_weights.shape}")
        # Load into model...
```

### 3. Check Before Access

Non-blocking checks for better control flow:

```python
# Check if ready for writing (no active readers)
if handler.is_ready(write=True):
    # Safe to update weights
    handler.put()
else:
    print("Readers are active, waiting...")

# Check if ready for reading (data is available)
if handler.is_ready(write=False, timeout_ms=100):
    with handler.get() as weights:
        # Use weights
        pass
else:
    print("No new data available yet")
```

### 4. Training-Inference Coordination

Complete example of coordinating training and inference:

```python
# train.py
import torch
from etha import TensorBus

bus = TensorBus()
model_weights = torch.randn(1024, 1024)
handler = bus.register_pair("model_v1", model_weights)

def train():
    for epoch in range(100):
        # Training logic...
        loss = train_one_epoch()
        
        # Wait for inference to finish reading
        if handler.wait_ready(write=True, timeout_ms=5000):
            # Update and publish new weights
            handler.put()
            print(f"Epoch {epoch}: Loss {loss:.4f}, weights published")
        else:
            print("Warning: Inference taking too long")

# inference.py
import torch
from etha import TensorBus

bus = TensorBus()
dummy_weights = torch.zeros(1024, 1024)
handler = bus.register_pair("model_v1", dummy_weights)

def infer():
    while True:
        # Wait for new weights
        if handler.wait_ready(write=False, timeout_ms=10000):
            with handler.get() as new_weights:
                # Load new weights
                model.load_state_dict(new_weights)
                
                # Run inference batch
                results = model(input_batch)
                process_results(results)
        else:
            print("No weight update for 10s, using existing weights")
            # Continue with existing weights
```

## Advanced Features

### Distributed Transfer (Coming Soon)

For distributed training/inference scenarios:

```python
from torch.distributed._tensor import DeviceMesh, Shard

# Define device topology
mesh = DeviceMesh("cuda", torch.arange(4).reshape(2, 2))
placement = Shard(0)

# Register with distributed parameters
handler = bus.register_pair(
    "distributed_model",
    tensor,
    mesh=mesh,
    placement=placement
)
```

### State Monitoring

```python
# Non-blocking check
is_ready = handler.is_ready(write=False)

# Blocking wait with timeout
ready = handler.wait_ready(write=False, timeout_ms=5000)

if not ready:
    print("Timeout waiting for data")
    # Handle timeout...
```

## Best Practices

### 1. Always Use Context Managers

✅ **Good:**
```python
with handler.get() as weights:
    model.load_state_dict(weights)
# Lock automatically released
```

❌ **Bad:**
```python
# Manual lock management - error prone!
weights = handler.tensor
model.load_state_dict(weights)
```

### 2. Use Batch Operations for Multiple Tensors

✅ **Good:**
```python
batched = bus.create_batched_handler([h1, h2, h3])
batched.put()
```

❌ **Bad:**
```python
# Multiple individual operations - higher overhead
h1.put()
h2.put()
h3.put()
```

### 3. Set Appropriate Timeouts

```python
# For latency-sensitive inference
handler.wait_ready(write=False, timeout_ms=100)

# For training synchronization
handler.wait_ready(write=True, timeout_ms=5000)

# For debugging (block indefinitely)
handler.wait_ready(write=False, timeout_ms=-1)
```

### 4. Unique Pair Names

```python
# Use descriptive, unique names
handler1 = bus.register_pair("actor_network_weights", tensor1)
handler2 = bus.register_pair("critic_network_weights", tensor2)

# Avoid generic names that might collide
# handler = bus.register_pair("weights", tensor)  # Too generic
```

## Common Patterns

### Pattern 1: Async RL with Multiple Actors

```python
# Training node
handlers = []
for actor_id in range(num_actors):
    h = bus.register_pair(f"weights_v{actor_id}", init_weights)
    handlers.append(h)

batched = bus.create_batched_handler(handlers)

# Broadcast to all actors
batched.put()

# Each actor node
my_handler = bus.register_pair(f"weights_v{my_id}", init_weights)
with my_handler.get() as weights:
    actor.load_weights(weights)
```

### Pattern 2: Pipeline with Multiple Stages

```python
# Stage 1 -> Stage 2 -> Stage 3
stage1_out = bus.register_pair("stage1_output", tensor1)
stage2_out = bus.register_pair("stage2_output", tensor2)

# Stage 1 writes, Stage 2 reads
stage1_out.put()
with stage2_out.get() as data:
    result = process_stage2(data)
    
# Stage 2 writes, Stage 3 reads
stage2_out.put()
```

### Pattern 3: Weight Versioning

```python
version = 0
while training:
    version += 1
    handler = bus.register_pair(f"weights_v{version}", new_weights)
    handler.put()
    
    # Old versions automatically garbage collected
```

## Troubleshooting

### Issue: Deadlock

**Symptom:** Both processes hang indefinitely

**Solution:** Always use context managers and timeouts
```python
if handler.wait_ready(write=False, timeout_ms=5000):
    with handler.get() as weights:
        # ...
else:
    # Handle timeout
```

### Issue: Stale Data

**Symptom:** Reader getting old data

**Solution:** Check readiness before reading
```python
if handler.is_ready(write=False):
    with handler.get() as weights:
        # Guaranteed fresh data
```

### Issue: High Latency

**Symptom:** Transfer takes longer than expected

**Solution:** Use batch operations and check for unnecessary waits
```python
# Batch multiple tensors
batched = bus.create_batched_handler(handlers)
batched.put()  # Single operation
```

## Next Steps

- Read [DESIGN.md](DESIGN.md) for architecture details
- Check [examples/](examples/) for complete working examples
- Review [tests/test_tensor_bus.py](tests/test_tensor_bus.py) for usage patterns

## Getting Help

- Open an issue on GitHub
- Check existing issues for similar problems
- Read the full API documentation
