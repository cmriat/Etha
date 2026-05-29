# Distributed Communication Benchmark

Benchmark for comparing M2M vs Gather-Broadcast communication methods with torch profiler and memory snapshot support.

## Basic Usage

```bash
# Run basic benchmark
pixi run -- torchrun --nproc_per_node=8 benchmark.py
```

## Enabling Profiling

Edit the configuration variables at the top of `main()` function in `benchmark.py`:

```python
# ========================================
# PROFILING CONFIGURATION (modify here)
# ========================================
ENABLE_PROFILING = True  # Set to True to enable profiling
PROFILE_SHAPES = [(4096, 4096), (8192, 8192), (16384, 16384)]  # Shapes to profile
PROFILE_WARMUP = 2  # Warmup steps for profiling
PROFILE_ACTIVE = 3  # Active profiling steps
PROFILE_FREQ = 5  # Total frequency (warmup + active + wait)
ENABLE_MEMORY_SNAPSHOT = True  # Generate memory snapshot pickle files
EXTRA_MEMORY_SNAPSHOTS = False  # Generate additional snapshots per iteration
```

Then run:
```bash
pixi run -- torchrun --nproc_per_node=8 benchmark.py
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_PROFILING` | False | Enable torch profiler and memory snapshot |
| `PROFILE_SHAPES` | [(4096,4096), (8192,8192), (16384,16384)] | List of tensor shapes to profile |
| `PROFILE_WARMUP` | 2 | Number of warmup steps for profiling |
| `PROFILE_ACTIVE` | 3 | Number of active profiling steps |
| `PROFILE_FREQ` | 5 | Total profiler frequency (warmup + active + wait) |
| `ENABLE_MEMORY_SNAPSHOT` | False | Generate memory snapshot pickle files |
| `EXTRA_MEMORY_SNAPSHOTS` | False | Generate additional memory snapshots per iteration |

## Output Structure

When profiling is enabled, outputs are organized as:

```
results/
├── traces/
│   └── mesh_01_4_2_1_1_1_1_4_2/
│       └── shape_(4096, 4096)/
│           └── rank_0/
│               ├── m2m/
│               │   └── iteration_3/
│               │       ├── rank0_trace.pt.trace.json
│               │       └── rank0_memory_snapshot.pickle
│               └── gather_broadcast_comm/
│                   └── iteration_3/
│                       ├── rank0_trace.pt.trace.json
│                       └── rank0_memory_snapshot.pickle
└── memory_snapshots/
    └── mesh_01_4_2_1_1_1_1_4_2/
        └── shape_(4096, 4096)/
            └── rank_0/
                ├── rank0_memory_snapshot_step_2.pickle
                └── rank0_memory_snapshot_step_3.pickle
```

## Analyzing Results

### Chrome Trace Files
1. Open Chrome browser and navigate to `chrome://tracing`
2. Click "Load" and select the `.pt.trace.json` file
3. Analyze GPU kernels, communication patterns, and timeline

### Memory Snapshot Files
```python
import pickle
import torch

# Load memory snapshot
with open("rank0_memory_snapshot.pickle", "rb") as f:
    snapshot = pickle.load(f)

# Analyze memory usage (use PyTorch memory profiler tools)
from torch.cuda._memory_viz import profile_plot
profile_plot(snapshot)
```

---

# Test results

Tested on 16 GPUs across 2 nodes. Each mesh combination is run under three
source placements; target placement is fixed to
`[Replicate, Replicate, StridedShard(0, split=k), Shard(0)]`
where `k = target_mesh[3]`.

| Config | Source placement |
|---|---|
| `no_partial`   | `[Replicate, Shard(0), Replicate, Shard(1)]` |
| `replicate_dp` | `[Replicate, Replicate, Replicate, Shard(1)]` |
| `partial_dp`   | `[Replicate, Partial("sum"), Replicate, Shard(1)]` |

## Mesh 1 — (1, 1, 4, 2) → (4, 2, 1, 1)

| no_partial | replicate_dp | partial_dp |
|---|---|---|
| ![](./results/throughput_benchmark_mesh_01_1_1_4_2_4_2_1_1_no_partial.png) | ![](./results/throughput_benchmark_mesh_01_1_1_4_2_4_2_1_1_replicate_dp.png) | ![](./results/throughput_benchmark_mesh_01_1_1_4_2_4_2_1_1_partial_dp.png) |

## Mesh 2 — (8, 1, 1, 1) → (1, 2, 2, 2)

| no_partial | replicate_dp | partial_dp |
|---|---|---|
| ![](./results/throughput_benchmark_mesh_02_8_1_1_1_1_2_2_2_no_partial.png) | ![](./results/throughput_benchmark_mesh_02_8_1_1_1_1_2_2_2_replicate_dp.png) | ![](./results/throughput_benchmark_mesh_02_8_1_1_1_1_2_2_2_partial_dp.png) |

## Mesh 3 — (1, 2, 4, 1) → (8, 1, 1, 1)

| no_partial | replicate_dp | partial_dp |
|---|---|---|
| ![](./results/throughput_benchmark_mesh_03_1_2_4_1_8_1_1_1_no_partial.png) | ![](./results/throughput_benchmark_mesh_03_1_2_4_1_8_1_1_1_replicate_dp.png) | ![](./results/throughput_benchmark_mesh_03_1_2_4_1_8_1_1_1_partial_dp.png) |

## Mesh 4 — (2, 2, 2, 1) → (1, 1, 4, 2)

| no_partial | replicate_dp | partial_dp |
|---|---|---|
| ![](./results/throughput_benchmark_mesh_04_2_2_2_1_1_1_4_2_no_partial.png) | ![](./results/throughput_benchmark_mesh_04_2_2_2_1_1_1_4_2_replicate_dp.png) | ![](./results/throughput_benchmark_mesh_04_2_2_2_1_1_1_4_2_partial_dp.png) |

## Mesh 5 — (2, 4, 1, 1) → (1, 2, 4, 1)

| no_partial | replicate_dp | partial_dp |
|---|---|---|
| ![](./results/throughput_benchmark_mesh_05_2_4_1_1_1_2_4_1_no_partial.png) | ![](./results/throughput_benchmark_mesh_05_2_4_1_1_1_2_4_1_replicate_dp.png) | ![](./results/throughput_benchmark_mesh_05_2_4_1_1_1_2_4_1_partial_dp.png) |

## Mesh 6 — (4, 1, 1, 2) → (2, 1, 1, 4)

| no_partial | replicate_dp | partial_dp |
|---|---|---|
| ![](./results/throughput_benchmark_mesh_06_4_1_1_2_2_1_1_4_no_partial.png) | ![](./results/throughput_benchmark_mesh_06_4_1_1_2_2_1_1_4_replicate_dp.png) | ![](./results/throughput_benchmark_mesh_06_4_1_1_2_2_1_1_4_partial_dp.png) |

## Mesh 7 — (4, 1, 2, 1) → (1, 1, 1, 8)

| no_partial | replicate_dp | partial_dp |
|---|---|---|
| ![](./results/throughput_benchmark_mesh_07_4_1_2_1_1_1_1_8_no_partial.png) | ![](./results/throughput_benchmark_mesh_07_4_1_2_1_1_1_1_8_replicate_dp.png) | ![](./results/throughput_benchmark_mesh_07_4_1_2_1_1_1_1_8_partial_dp.png) |

## Mesh 8 — (4, 2, 1, 1) → (1, 1, 2, 4)

| no_partial | replicate_dp | partial_dp |
|---|---|---|
| ![](./results/throughput_benchmark_mesh_08_4_2_1_1_1_1_2_4_no_partial.png) | ![](./results/throughput_benchmark_mesh_08_4_2_1_1_1_1_2_4_replicate_dp.png) | ![](./results/throughput_benchmark_mesh_08_4_2_1_1_1_1_2_4_partial_dp.png) |

---

# KVStore Benchmark

Benchmark comparing EtcdStore vs TorchTCPStore performance for key-value operations.

## Basic Usage

```bash
# Run benchmark for both backends
pixi run -e dev python bench/kvstore_benchmark.py

# Run with more operations
pixi run -e dev python bench/kvstore_benchmark.py --num-ops 1000

# Run only etcd backend
pixi run -e dev python bench/kvstore_benchmark.py --backend etcd

# Run only TCPStore backend
pixi run -e dev python bench/kvstore_benchmark.py --backend tcp

# Show help
pixi run -e dev python bench/kvstore_benchmark.py --help
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--etcd-host` | localhost | etcd server host |
| `--etcd-port` | 2379 | etcd server port |
| `--tcp-host` | localhost | TCPStore server host |
| `--tcp-port` | 29501 | TCPStore server port |
| `--num-ops` | 500 | Number of operations for benchmarks |
| `--backend` | both | Which backend to benchmark (both/etcd/tcp) |

## Benchmarked Operations

| Operation | Description |
|-----------|-------------|
| `set` | Set string key-value pairs |
| `get` | Get string values by key |
| `exists` | Check if key exists |
| `set_bytes` | Set binary data (64B, 1KB, 4KB) |
| `get_bytes` | Get binary data |
| `wait_for_key(existing)` | Wait for key that already exists |
| `wait_for_key(async)` | Wait for key written after delay |
| `wait_for_keys(N)` | Wait for N keys matching pattern |

## Notes

- **etcd** requires a running etcd server (default: localhost:2379)
- **TorchTCPStore** starts its own server automatically
- EtcdStore uses watch mechanism for `wait_for_key` (efficient)
- TorchTCPStore uses polling for `wait_for_key` (less efficient but no external dependency)