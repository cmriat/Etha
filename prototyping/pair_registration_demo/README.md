# Pair Registration Demo - Distributed Tensor Bus

This prototype demonstrates the full end-to-end pair registration flow with:
- 8 Agent processes forming a torch.distributed NCCL group
- 4 Inference workers (simulated, local_name="inference")
- 4 Training workers (simulated, local_name="training")
- TCPStore-based pair matching
- State LMDB for Worker-Agent communication

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ TCPStore (127.0.0.1:29500)                                      │
│ - Metadata exchange for pair matching                           │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                              │ (all_gather_object + polling)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Agent Process Group (NCCL, world_size=8)                        │
│                                                                 │
│  Rank 0-3 (inference side)     Rank 4-7 (training side)         │
│  ├─ agent_rank0               ├─ agent_rank4                    │
│  ├─ agent_rank1               ├─ agent_rank5                    │
│  ├─ agent_rank2               ├─ agent_rank6                    │
│  └─ agent_rank3               └─ agent_rank7                    │
│                                                                 │
│  Each agent has:                                                │
│  - CommandQueue LMDB: receive RegisterPair from Worker          │
│  - State LMDB: publish PairState to Worker                      │
└─────────────────────────────────────────────────────────────────┘
         ↑                                        ↑
         │ LMDB IPC                               │ LMDB IPC
         │                                        │
┌─────────────────────┐              ┌────────────────────────┐
│ Inference Workers   │              │ Training Workers       │
│ (4 simulated)       │              │ (4 simulated)          │
│                     │              │                        │
│ Worker 0 → Agent 0 │              │ Worker 0 → Agent 4    │
│ Worker 1 → Agent 1 │              │ Worker 1 → Agent 5    │
│ Worker 2 → Agent 2 │              │ Worker 2 → Agent 6    │
│ Worker 3 → Agent 3 │              │ Worker 3 → Agent 7    │
│                     │              │                        │
│ local_name="inference" │           │ local_name="training"  │
│ pair_name="obs"     │              │ pair_name="obs"        │
└─────────────────────┘              └────────────────────────┘
```

## Message Flow

```
1. Worker → Agent (CommandQueue LMDB)
   - Worker enqueues RegisterPair(pair_name="obs", local_name="inference", remote_name="training")

2. Agent Processing (agent.py)
   - Dequeue RegisterPair
   - Write to TCPStore: pair:obs:inference:rank{i} = "1"
   - Poll TCPStore until local peer complete (ranks 0-3)
   - Poll TCPStore until remote peer "training" complete (ranks 4-7)
   - Create PairState(local_ranks=[0,1,2,3], remote_ranks=[4,5,6,7])

3. Agent → Worker (State LMDB)
   - Write PairState to State LMDB

4. Worker Verification (client.py)
   - Poll State LMDB until PairState.status == "matched"
   - Return PairHandler to user
```

## Components

### 1. `shared.py`
Shared constants between all processes:
- `PAIR_NAME = "obs"`
- `TCPSTORE_HOST` and `TCPSTORE_PORT`
- Helper functions: `get_agent_command_queue_path(rank)`, `get_agent_state_path(rank)`

### 2. `agent.py`
TensorBusAgent launcher (requires torchrun):
- Initializes torch.distributed (NCCL backend)
- Creates CommandQueue and State LMDB for this rank
- Enters main loop: poll CommandQueue → handle RegisterPair → write PairState

### 3. `worker_inference.py`
4 inference worker processes (requires torchrun):
- Each worker (local_rank 0-3) maps to agent ranks 0-3
- Registers pair "obs" with local_name="inference"
- Polls State LMDB until matched
- Prints verification result

### 4. `worker_training.py`
4 training worker processes (requires torchrun):
- Each worker (local_rank 0-3) maps to agent ranks 4-7
- Registers pair "obs" with local_name="training"
- Polls State LMDB until matched
- Prints verification result

## Usage

### Prerequisites

Ensure you have 8 GPUs available (or modify code to use fewer GPUs for testing).

### Step 1: Start Agent Process Group

```bash
# Terminal 1
torchrun --nproc_per_node=8 prototyping/pair_registration_demo/agent.py
```

**Expected output:**
```
[Agent 0] Initialized successfully
[Agent 0] Entering main loop (polling for commands)...
[Agent 1] Initialized successfully
...
```

### Step 2: Start Inference Workers

```bash
# Terminal 2
torchrun --nproc_per_node=4 --master_port=29501 prototyping/pair_registration_demo/worker_inference.py
```

**Note**: `--master_port=29501` avoids conflict with Agent's TCPStore on port 29500

**Expected output:**
```
Inference Worker (local_rank=0) starting...
Worker 0: Registering pair 'obs' as 'inference'
...
(waiting for training workers to register)
```

### Step 3: Start Training Workers

```bash
# Terminal 3
torchrun --nproc_per_node=4 --master_port=29502 prototyping/pair_registration_demo/worker_training.py
```

**Note**: `--master_port=29502` avoids conflict with other processes

**Expected output:**
```
Training Worker (local_rank=0) starting...
Worker 0: Registering pair 'obs' as 'training'
...
```

### Step 4: Verify Matching

Once both sides register, you should see in **Terminal 1 (Agent)**:

```
[Agent 0] Pair 'obs' matched! Local 'inference': [0, 1, 2, 3], Remote 'training': [4, 5, 6, 7]
[Agent 1] Pair 'obs' matched! Local 'inference': [0, 1, 2, 3], Remote 'training': [4, 5, 6, 7]
...
[Agent 4] Pair 'obs' matched! Local 'training': [4, 5, 6, 7], Remote 'inference': [0, 1, 2, 3]
...
```

And in **Terminal 2 (Inference Workers)**:

```
✅ Worker 0: Pair 'obs' matched!
   Elapsed time: 1.23s
✅ Worker 1: Pair 'obs' matched!
...
```

And in **Terminal 3 (Training Workers)**:

```
✅ Worker 0: Pair 'obs' matched!
   Elapsed time: 1.15s
✅ Worker 1: Pair 'obs' matched!
...
```

## Cleanup

```bash
# Stop all processes (Ctrl+C in each terminal)

# Remove LMDB files
rm /tmp/agent_rank*_command.lmdb*
rm /tmp/agent_rank*_state.lmdb*
```

## What This Demonstrates

✅ **Agent Process Group**: 8 ranks forming a torch.distributed NCCL group
✅ **TCPStore Pair Matching**: Decentralized discovery of my_ranks and remote_ranks
✅ **Worker-Agent IPC**: CommandQueue (Worker→Agent) + State LMDB (Agent→Worker)
✅ **State Verification**: Workers poll State LMDB to confirm pair matching
✅ **End-to-End Flow**: Full lifecycle from Worker registration to verified matching

## Implementation Details

### Key Design Decisions

1. **Explicit Remote Peer**: Users must specify `remote_name` when registering (no hardcoded peer names)
2. **Decentralized Matching**: No leader election, every Agent independently discovers ranks via TCPStore
3. **Peer-to-Peer Model**: No fixed writer/reader roles, both peers can send() or recv()
4. **State Persistence**: PairState stored in LMDB allows Workers to verify asynchronously
5. **Polling-Based Sync**: Workers poll State LMDB every 100ms until matched (simple, robust)

### File Structure

```
prototyping/pair_registration_demo/
├── README.md                    # This file
├── shared.py                    # Shared constants and utilities
├── agent.py                    # TensorBusAgent launcher (torchrun)
├── worker_inference.py          # Simulates 4 inference workers
└── worker_training.py           # Simulates 4 training workers
```

### Code Modifications

This prototype required modifications to:
- `src/etha/tensor_bus/agent.py`: Added `lmdb_state_path` parameter and PairState persistence
- `src/etha/tensor_bus/client.py`: Added `agent_state_lmdb_path` parameter and state polling

## Next Steps

After validating this prototype, the next steps are:
1. Implement actual tensor send/recv operations (using p2p_communicate)
2. Add DeviceMesh and Placement metadata to PairState
3. Implement CommHandle for async operations
4. Add error handling and timeout logic
5. Performance optimization (reduce polling overhead)

## Troubleshooting

**Issue**: `torch.distributed.DistNetworkError: address already in use` on port 29500
**Fix**: This happens when worker torchrun tries to use port 29500 (already used by Agent). Use `--master_port` flag:
```bash
torchrun --nproc_per_node=4 --master_port=29501 worker_inference.py
torchrun --nproc_per_node=4 --master_port=29502 worker_training.py
```

**Issue**: `RuntimeError: Address already in use` from Agent
**Fix**: Kill existing TCPStore master process or change `TCPSTORE_PORT` in `shared.py`

**Issue**: Workers timeout waiting for pair matching
**Fix**: Ensure Agent is running and check logs for errors

**Issue**: CUDA out of memory
**Fix**: Reduce number of GPUs or use smaller tensors

**Issue**: `lmdb.DbsFullError: MDB_DBS_FULL: Environment maxdbs limit reached`
**Fix**: This happens when using old LMDB files created with different `max_dbs`. Clean up and restart:
```bash
rm /tmp/agent_rank*_state.lmdb*
# Then restart Agent
```

**Issue**: LMDB lock errors
**Fix**: Run cleanup commands to remove stale LMDB lock files
