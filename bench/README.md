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
│               ├── chunk_comm/
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
Test on 16 GPUs with 2 nodes.

source mesh placement: [Replicate(), Replicate(), Shard(0), Shard(1)]
target mesh placement: [Replicate(), _StridedShard(1, split_factor=2), Replicate(), Shard(1)]

## Mesh Configuration 1
* 1, 1, 4, 2 -> 4, 2, 1, 1
![Mesh 1](./results/throughput_benchmark_mesh_01_1_1_4_2_4_2_1_1.png)

## Mesh Configuration 2
* 8, 1, 1, 1 -> 1, 2, 2, 2
![Mesh 2](./results/throughput_benchmark_mesh_02_8_1_1_1_1_2_2_2.png)

## Mesh Configuration 3
* 1, 2, 4, 1 -> 8, 1, 1, 1
![Mesh 3](./results/throughput_benchmark_mesh_03_1_2_4_1_8_1_1_1.png)

## Mesh Configuration 4
* 2, 2, 2, 1 -> 1, 1, 4, 2
![Mesh 4](./results/throughput_benchmark_mesh_04_2_2_2_1_1_1_4_2.png)

## Mesh Configuration 5
* 2, 4, 1, 1 -> 1, 2, 4, 1
![Mesh 5](./results/throughput_benchmark_mesh_05_2_4_1_1_1_2_4_1.png)

## Mesh Configuration 6
* 4, 1, 1, 2 -> 2, 1, 1, 4
![Mesh 6](./results/throughput_benchmark_mesh_06_4_1_1_2_2_1_1_4.png)

## Mesh Configuration 7
* 4, 1, 2, 1 -> 1, 1, 1, 8
![Mesh 7](./results/throughput_benchmark_mesh_07_4_1_2_1_1_1_1_8.png)

## Mesh Configuration 8
* 4, 2, 1, 1 -> 1, 1, 2, 4
![Mesh 8](./results/throughput_benchmark_mesh_08_4_2_1_1_1_1_2_4.png)