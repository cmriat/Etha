# CommandQueue Prototype - Zero-Copy Tensor Sharing

This prototype demonstrates zero-copy tensor sharing between processes using the CommandQueue.

## Components

- **shared.py**: LMDB storage utilities for tensor payloads
- **writer.py**: Creates tensor, sends RegisterTensor message, modifies values
- **reader.py**: Receives message, rebuilds tensor, monitors changes

## Message Flow

### RegisterTensor Message (Simplified)
```python
RegisterTensor(
    tensor_id="tensor_0",          # Unique identifier
    storage_key="tensor_0",        # LMDB key for pickled payload
    writer_pid=12345,              # For ptrace authorization
    timestamp=1234567890.0         # Inherited from Command base
)
```

**Note:** Tensor metadata (shape, dtype, device, CUDA pointer) is stored in the pickled
payload at `storage_key`, not in the message. This avoids redundancy and trusts
PyTorch's ForkingPickler to handle all serialization.

## Usage

### Terminal 1: Start Writer

```bash
pixi run -e dev python prototyping/command_queue_prototype/writer.py
```

The writer will:
1. Create a CUDA tensor with 10 float32 elements (all zeros)
2. Store pickled tensor in LMDB
3. Send RegisterTensor via CommandQueue
4. Wait for you to start the reader
5. Continuously modify tensor values (0, 1, 2, ...)

### Terminal 2: Start Reader

```bash
pixi run -e dev python prototyping/command_queue_prototype/reader.py
```

The reader will:
1. Dequeue RegisterTensor message from CommandQueue
2. Load pickled tensor from LMDB
3. Rebuild tensor (zero-copy via ForkingPickler)
4. Monitor tensor values in a loop

**Expected Output:**
- Reader should see tensor values changing: [0,0,0,...] → [1,1,1,...] → [2,2,2,...]
- CUDA pointer should match between writer and reader
- This confirms zero-copy is working!

## Cleanup

```bash
# Remove LMDB files
rm /tmp/tensor_storage.lmdb*
rm /tmp/tensor_queue.lmdb*
```

## Key Differences from dev/lmdb/ Prototype

1. **Command Channel**: Uses CommandQueue instead of direct LMDB key-value
2. **Type Safety**: RegisterTensor message with schema validation
3. **Separation**: Command queue and tensor storage use separate LMDB files
4. **Metadata**: RegisterTensor carries rich metadata (shape, dtype, device, etc.)

## What This Demonstrates

✅ CommandQueue successfully passes messages between processes
✅ msgspec TaggedUnion correctly serializes/deserializes RegisterTensor
✅ Zero-copy tensor sharing still works with the new architecture
✅ Foundation ready for building full Tensor Bus server/client

## Next Steps

1. Implement Tensor Bus Server that continuously processes commands
2. Implement Host Client with high-level API
3. Add more message types (UpdateTensor, DeleteTensor, etc.)
4. Add error handling and resource cleanup
