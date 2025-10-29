# CommandQueue Prototype - Zero-Copy Tensor Sharing

This prototype demonstrates zero-copy tensor sharing between processes using the CommandQueue system.

## Components

- **shared.py**: LMDB path constants for command queue
- **writer.py**: Creates CUDA tensor, sends RegisterTensor message with embedded payload, modifies values
- **reader.py**: Receives message, rebuilds tensor via ForkingPickler, monitors changes

## Message Flow

### RegisterTensor Message
```python
RegisterTensor(
    pair_name="pair_0",              # Process pair identifier
    tensor_name="tensor_0",          # Unique tensor identifier
    tensor_payload=b"...",           # Pickled tensor data
    timestamp=1234567890.0           # Message timestamp
)
```


## Usage

### Terminal 1: Start Writer

```bash
pixi run -e dev python prototyping/command_queue_prototype/writer.py
```

The writer will:
1. Create a CUDA tensor with 10 float32 elements (all zeros)
2. Serialize tensor using PyTorch's ForkingPickler
3. Send RegisterTensor message with embedded payload via CommandQueue
4. Wait for you to start the reader
5. Continuously modify tensor values (0, 1, 2, ...)

### Terminal 2: Start Reader

```bash
pixi run -e dev python prototyping/command_queue_prototype/reader.py
```

The reader will:
1. Dequeue RegisterTensor message from CommandQueue
2. Extract embedded tensor payload from message
3. Rebuild tensor (zero-copy via ForkingPickler)
4. Monitor tensor values in a loop

**Expected Output:**
- Reader should see tensor values changing: [0,0,0,...] → [1,1,1,...] → [2,2,2,...]
- CUDA pointer should match between writer and reader
- This confirms zero-copy is working!

## Cleanup

```bash
# Remove LMDB files
rm /tmp/tensor_queue.lmdb*
```


## What This Demonstrates

✅ CommandQueue successfully passes messages with embedded tensor payloads
✅ msgspec correctly serializes/deserializes RegisterTensor with binary data
✅ Zero-copy tensor sharing works with simplified architecture
✅ Foundation ready for building full Tensor Bus server/client

## Next Steps

1. Implement Tensor Bus Server for continuous command processing
2. Implement Host Client with high-level API
3. Add more message types (UpdateTensor, DeleteTensor, etc.)
4. Add error handling and resource cleanup
5. Scale to multiple tensor pairs and processes