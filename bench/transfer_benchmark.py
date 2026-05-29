"""Distributed communication benchmark for M2M vs Gather-Broadcast methods."""

import os
import math
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.distributed as dist
from upath import UPath
from utils import ProfilingSpec, dump_memory_snapshot, maybe_enable_profiling
from torch.distributed._tensor import Shard, Replicate, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Partial, _StridedShard

from etha.comm import (
    bucket_comm,
    get_m2m_map,
    m2m_to_chunks,
    chunk_to_bucket_ops,
    gather_broadcast_comm,
)
from etha.pg_utils import get_or_create_process_group
from etha.comm.utils import enumerate_partial_subgroup_ranks

# Hardware bandwidth parameters (GB/s)
RDMA_SEND_BANDWIDTH_GB_S = 50.0
NVLINK_Single_BANDWIDTH_GB_S = 450.0
BUCKET_SIZE_BYTES = 256 * 1024 * 1024


def calculate_transfer_bytes(
    target_mesh: DeviceMesh,
    target_placements: tuple,
    total_bytes: int,
) -> int:
    """Calculate ideal bytes that need to be received by all target ranks.

    Start from origin tensor size, then multiply by replication factors.
    - Replicate(): data is duplicated → multiply by mesh_dim_size
    - Shard(): data is partitioned → no change to total

    Returns:
        Total bytes all target ranks need to receive (sum across all ranks).
    """
    for i, placement in enumerate(target_placements):
        mesh_dim_size = target_mesh.mesh.shape[i]
        if isinstance(placement, Replicate):
            # Data is replicated across this mesh dimension
            total_bytes *= mesh_dim_size

    return total_bytes


def calculate_ideal_bandwidth(
    source_mesh: DeviceMesh,
    target_mesh: DeviceMesh,
    target_placements: tuple,
    origin_tensor_bytes: int,
    gpus_per_node: int,
) -> float:
    """Calculate ideal bandwidth upper bound.

    Two-stage model:
    1. IB stage: send unique data via parallel IB links
    2. NVLink stage: all target nodes do intra-node all-gather in parallel

    Returns:
        Ideal bandwidth in GB/s
    """
    source_ranks = source_mesh.mesh.flatten().tolist()
    target_ranks = target_mesh.mesh.flatten().tolist()

    # Count unique nodes
    source_nodes = set(r // gpus_per_node for r in source_ranks)
    target_nodes = set(r // gpus_per_node for r in target_ranks)

    num_source_nodes = len(source_nodes)
    num_target_nodes = len(target_nodes)

    # Parallel RDMA links (upper bound)
    parallel_links = min(num_source_nodes, num_target_nodes) * gpus_per_node
    total_ib_bandwidth = RDMA_SEND_BANDWIDTH_GB_S * parallel_links

    # Total transfer requirement (including replicate)
    total_transfer_gb = calculate_transfer_bytes(target_mesh, target_placements, origin_tensor_bytes) / (1024**3)

    # Unique data (one copy of full tensor)
    unique_data_gb = origin_tensor_bytes / (1024**3)

    # Replicate data (additional copies needed)
    replicate_data_gb = total_transfer_gb - unique_data_gb

    # Stage 1: IB transfer time
    ib_time = unique_data_gb / total_ib_bandwidth if total_ib_bandwidth > 0 else float("inf")

    # Stage 2: NVLink all-gather time (all target nodes work in parallel)
    total_nvlink_bandwidth = num_target_nodes * gpus_per_node * NVLINK_Single_BANDWIDTH_GB_S
    nvlink_time = replicate_data_gb / total_nvlink_bandwidth if total_nvlink_bandwidth > 0 else 0

    # Total time
    total_time = max(ib_time, nvlink_time)

    # Ideal bandwidth
    ideal_bandwidth = total_transfer_gb / total_time if total_time > 0 else float("inf")

    return ideal_bandwidth


def benchmark_single_shape(
    origin_tensors: list[torch.Tensor],  # Singleton list of the baseline reference tensor
    source_dist_tensors: list,  # List of source distributed tensors
    target_local_tensors: list,  # List of target local tensors
    num_tensors: int,  # Logical batch size (origin_tensors holds only index 0 to save memory)
    shape: tuple,
    chunks: list,  # All chunks from all tensors (extended)
    buckets: list,
    current_source_mesh: DeviceMesh,
    current_target_mesh: DeviceMesh,
    source_specs: list,
    target_specs: list,
    source_world_size: int,
    device: str,
    rank: int,
    local_rank: int,  # noqa
    warmup_iter: int,
    profile_iter: int,
    gpus_per_node: int,
    profiling_config: dict | None = None,
    mesh_info: str = "",
    has_partial: bool = False,
):
    """Benchmark M2M vs Gather-Broadcast for a batch of tensors.

    Tests batch transfer of multiple tensors with unified chunk list.

    Args:
        ... (existing args) ...
        profiling_config: Dict with profiling settings if enabled
        mesh_info: String identifier for current mesh configuration

    Returns:
        dict with benchmark results if rank == 0, None otherwise.
    """
    # Calculate total size of all tensors in batch (per-tensor size × num_tensors).
    total_tensor_size_bytes = origin_tensors[0].nelement() * origin_tensors[0].element_size() * num_tensors

    # Check if we need to profile this shape
    should_profile = profiling_config is not None and shape in profiling_config["profile_shapes"]

    # M2M method = single-entry buckets (no coalescing); built once, reused.
    m2m_buckets = chunk_to_bucket_ops(chunks=chunks, bucket_size=1)

    # M2M method warmup
    dist.barrier()
    for _ in range(warmup_iter):
        bucket_comm(buckets=m2m_buckets)
    if device == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    # M2M method with profiling if enabled
    if should_profile:
        m2m_profiling_spec = ProfilingSpec(
            enable_profiling=True,
            dump_folder=UPath(profiling_config["dump_folder"]),
            save_traces_folder=UPath(f"traces/{mesh_info}/shape_{shape[0]}/rank_{rank}/m2m"),
            profile_freq=profiling_config["profile_freq"],
            warmup_steps=profiling_config["warmup_steps"],
            active_steps=profiling_config["active_steps"],
            enable_memory_snapshot=profiling_config["enable_memory_snapshot"],
        )

        with maybe_enable_profiling(m2m_profiling_spec, global_step=0) as profiler:
            if profiler is not None:
                # Profiled iterations
                for step in range(m2m_profiling_spec.profile_freq):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    dist.barrier()
                    profiler.step()
                    for _ in range(10):
                        bucket_comm(buckets=m2m_buckets)

                    # Memory snapshot if requested
                    if m2m_profiling_spec.enable_memory_snapshot and step == m2m_profiling_spec.warmup_steps + 1:
                        snapshot_dir = f"{m2m_profiling_spec.dump_folder}/memory_snapshots/{mesh_info}/shape_{shape[0]}/rank_{rank}"
                        dump_memory_snapshot(snapshot_dir, step, rank)
            else:
                pass

    # Time measurement with CUDA events
    if device == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(profile_iter):
        bucket_comm(buckets=m2m_buckets)
    end_event.record()

    torch.cuda.synchronize()
    m2m_time = start_event.elapsed_time(end_event) / 1000.0 / profile_iter  # ms -> s

    dist.barrier()
    for _ in range(warmup_iter):
        bucket_comm(buckets=buckets)
    if device == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    if should_profile:
        bucket_profiling_spec = ProfilingSpec(
            enable_profiling=True,
            dump_folder=UPath(profiling_config["dump_folder"]),
            save_traces_folder=UPath(f"traces/{mesh_info}/shape_{shape[0]}/rank_{rank}/bucket_comm"),
            profile_freq=profiling_config["profile_freq"],
            warmup_steps=profiling_config["warmup_steps"],
            active_steps=profiling_config["active_steps"],
            enable_memory_snapshot=profiling_config["enable_memory_snapshot"],
        )

        with maybe_enable_profiling(bucket_profiling_spec, global_step=0) as profiler:
            if profiler is not None:
                for step in range(bucket_profiling_spec.profile_freq):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    dist.barrier()
                    profiler.step()
                    for _ in range(10):
                        bucket_comm(buckets=buckets)
                    if bucket_profiling_spec.enable_memory_snapshot and step == bucket_profiling_spec.warmup_steps + 1:
                        snapshot_dir = f"{bucket_profiling_spec.dump_folder}/memory_snapshots/{mesh_info}/shape_{shape[0]}/rank_{rank}"
                        dump_memory_snapshot(snapshot_dir, step + 2000, rank)
            else:
                pass

    # Time measurement with CUDA events
    if device == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(profile_iter):
        bucket_comm(buckets=buckets)
    end_event.record()

    torch.cuda.synchronize()
    bucket_time = start_event.elapsed_time(end_event) / 1000.0 / profile_iter  # ms -> s

    # Gather-Broadcast method warmup (only test first tensor as reference).
    # For Partial source: user manually redistributes to Replicate first (inside
    # the timed loop so the reduce cost is counted), then calls gather_broadcast.
    def _baseline_iter():
        if source_dist_tensors:
            src_dt = source_dist_tensors[0]
            if has_partial:
                src_dt = _pre_reduce_to_replicate(src_dt, current_source_mesh, source_specs)
            return gather_broadcast_comm(
                current_target_mesh, target_specs, src_dt, origin_tensors[0], source_world_size
            )
        return gather_broadcast_comm(current_target_mesh, target_specs, None, origin_tensors[0], source_world_size)

    dist.barrier()
    bc_result = None
    for _ in range(warmup_iter):
        bc_result = _baseline_iter()

    # Verify correctness for first tensor (M2M modifies target tensors in-place).
    if target_local_tensors and target_local_tensors[0] is not None and bc_result is not None:
        if not torch.allclose(target_local_tensors[0], bc_result.to_local()):
            print(f"[Rank {rank}] M2M result shape: {target_local_tensors[0].shape}")
            print(f"[Rank {rank}] Baseline result shape: {bc_result.to_local().shape}")
            print(f"[Rank {rank}] Max diff: {(target_local_tensors[0] - bc_result.to_local()).abs().max().item()}")
            print(f"[Rank {rank}] M2M sample: {target_local_tensors[0]}")
            print(f"[Rank {rank}] Baseline sample: {bc_result.to_local()}")
            raise ValueError("M2M and Gather-Broadcast results mismatch!")

    if device == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    # Gather-Broadcast method with profiling if enabled
    if should_profile:
        # Setup profiling for Gather-Broadcast
        gb_profiling_spec = ProfilingSpec(
            enable_profiling=True,
            dump_folder=UPath(profiling_config["dump_folder"]),
            save_traces_folder=UPath(f"traces/{mesh_info}/shape_{shape[0]}/rank_{rank}/gather_broadcast_comm"),
            profile_freq=profiling_config["profile_freq"],
            warmup_steps=profiling_config["warmup_steps"],
            active_steps=profiling_config["active_steps"],
            enable_memory_snapshot=profiling_config["enable_memory_snapshot"],
        )

        with maybe_enable_profiling(gb_profiling_spec, global_step=0) as profiler:
            if profiler is not None:
                # Profiled iterations
                for step in range(gb_profiling_spec.profile_freq):
                    profiler.step()
                    for _ in range(10):
                        _baseline_iter()

                    # Additional memory snapshot if requested
                    if gb_profiling_spec.enable_memory_snapshot and step == gb_profiling_spec.warmup_steps + 1:
                        snapshot_dir = (
                            f"{gb_profiling_spec.dump_folder}/memory_snapshots/{mesh_info}/shape_{shape[0]}/rank_{rank}"
                        )
                        dump_memory_snapshot(snapshot_dir, step + 1000, rank)  # +1000 to distinguish from M2M
            else:
                pass

    # Time measurement with CUDA events
    if device == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(profile_iter):
        _baseline_iter()
    end_event.record()

    torch.cuda.synchronize()
    baseline_time = start_event.elapsed_time(end_event) / 1000.0 / profile_iter  # ms -> s

    if rank == 0:
        # Calculate ideal bytes that target ranks need to receive (for all tensors in batch)
        ideal_transfer_bytes = calculate_transfer_bytes(current_target_mesh, target_specs, total_tensor_size_bytes)
        ideal_transfer_gb = ideal_transfer_bytes / (1024**3)

        # Calculate ideal bandwidth (hardware topology aware)
        ideal_bw = calculate_ideal_bandwidth(
            current_source_mesh,
            current_target_mesh,
            target_specs,
            total_tensor_size_bytes,
            gpus_per_node,
        )

        # Effective throughput: how fast we complete the redistribution task
        # M2M: tests all tensors in batch
        m2m_effective_throughput = ideal_transfer_gb / m2m_time
        bucket_effective_throughput = ideal_transfer_gb / bucket_time

        # Baseline: only tests 1 tensor, so divide by num_tensors
        single_tensor_ideal_transfer_gb = ideal_transfer_gb / num_tensors
        baseline_effective_throughput = single_tensor_ideal_transfer_gb / baseline_time

        print(f"    Batch: {num_tensors} tensors x {shape} = {total_tensor_size_bytes / (1024**2):.2f} MB total")
        print(f"    Ideal M2M transfer (target needs): {ideal_transfer_bytes / (1024**2):.2f} MB")
        print(f"    Ideal bandwidth (RDMA+NVLink): {ideal_bw:.2f} GB/s")
        print(f"    M2M batch effective throughput ({num_tensors} tensors): {m2m_effective_throughput:.2f} GB/s")
        print(f"    Bucket batch effective throughput ({num_tensors} tensors): {bucket_effective_throughput:.2f} GB/s")
        print(f"    Baseline effective throughput (1 tensor): {baseline_effective_throughput:.2f} GB/s")

        return {
            "tensor_shape": str(shape),
            "num_tensors": num_tensors,
            "tensor_size_mb": total_tensor_size_bytes / (1024**2),
            "ideal_transfer_mb": ideal_transfer_bytes / (1024**2),
            "ideal_bandwidth_gb_s": ideal_bw,
            "m2m_effective_throughput_gb_s": m2m_effective_throughput,
            "bucket_effective_throughput_gb_s": bucket_effective_throughput,
            "baseline_effective_throughput_gb_s": baseline_effective_throughput,
        }

    return None


def _placements_to_str(placements: list) -> str:
    parts = []
    for placement in placements:
        # _StridedShard subclasses Shard; match its case first.
        match placement:
            case Replicate():
                parts.append("Replicate")
            case _StridedShard():
                parts.append(f"StridedShard({placement.dim}, split={placement.split_factor})")
            case Shard():
                parts.append(f"Shard({placement.dim})")
            case Partial():
                parts.append(f"Partial({placement.reduce_op})")
            case _:
                parts.append(str(placement))
    return "[" + ", ".join(parts) + "]"


def _build_source_partial_groups(
    source_mesh: DeviceMesh,
    source_placements: list,
    this_rank: int,
    source_ranks: list[int],
) -> list[tuple[dist.ProcessGroup, str]] | None:
    """Mirror of agent._create_partial_groups for benchmark use."""
    if not any(isinstance(p, Partial) for p in source_placements):
        return None
    full_set = set(source_ranks)
    full_source_group = get_or_create_process_group(source_ranks)
    my_groups: list[tuple[dist.ProcessGroup, str]] = []
    for mesh_dim_idx, p in enumerate(source_placements):
        if not isinstance(p, Partial):
            continue
        for sub_ranks in enumerate_partial_subgroup_ranks(source_mesh.mesh, mesh_dim_idx):
            if set(sub_ranks) == full_set:
                group = full_source_group
            else:
                group = get_or_create_process_group(sub_ranks)
            if this_rank in sub_ranks:
                my_groups.append((group, p.reduce_op))
    return my_groups or None


def _pre_reduce_to_replicate(source_dt, source_mesh: DeviceMesh, source_placements: list):
    """The baseline's manual Partial→Replicate step (timed inline for fairness)."""
    effective = [Replicate() if isinstance(p, Partial) else p for p in source_placements]
    return source_dt.redistribute(source_mesh, effective)


def generate_result_plot(
    results: list,
    source_mesh_shape: tuple,
    target_mesh_shape: tuple,
    mesh_idx: int,
    num_tensors_per_batch: int,
    source_specs: list,
    target_specs: list,
    config_name: str = "",
    has_partial: bool = False,
):
    """Generate and save throughput comparison plot.

    Partial configs draw the baseline line as "Pre-reduce + Gather-Broadcast"
    since the reduce cost is timed inside the baseline loop.
    """
    df = pd.DataFrame(results)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    baseline_label = "Pre-reduce + Gather-Broadcast" if has_partial else "Gather-Broadcast"
    chunk_label = "Chunk-Based (Partial in-pipeline)" if has_partial else "Chunk-Based"

    plot_configs = [
        ("ideal_bandwidth_gb_s", "Ideal (RDMA+NVLink)", "--", "green", None),
        ("m2m_effective_throughput_gb_s", chunk_label, "o", "blue", 8),
        ("bucket_effective_throughput_gb_s", "Bucket-Based", "s", "purple", 7),
        ("baseline_effective_throughput_gb_s", baseline_label, "X", "orange", 8),
    ]

    for y_col, label, marker, color, markersize in plot_configs:
        if marker == "--":
            ax.plot(df["tensor_size_mb"], df[y_col], linestyle="--", color=color, linewidth=2, label=label, alpha=0.7)
        else:
            sns.lineplot(
                data=df,
                x="tensor_size_mb",
                y=y_col,
                marker=marker,
                markersize=markersize,
                label=label,
                ax=ax,
                color=color,
            )

    title_extra = f" [{config_name}]" if config_name else ""
    ax.set_title(
        f"Batch Transfer Throughput ({num_tensors_per_batch} tensors){title_extra}\n"
        f"Mesh: {source_mesh_shape} → {target_mesh_shape}\n"
        f"source_specs={_placements_to_str(source_specs)}\n"
        f"target_specs={_placements_to_str(target_specs)}",
        fontsize=11,
        weight="bold",
    )
    ax.set_xlabel("Total Batch Size (MB)", fontsize=11)
    ax.set_ylabel("Effective Throughput (GB/s)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    src = "_".join(map(str, source_mesh_shape))
    tgt = "_".join(map(str, target_mesh_shape))
    suffix = f"_{config_name}" if config_name else ""
    fig_path = RESULTS_DIR / f"throughput_benchmark_mesh_{mesh_idx + 1:02d}_{src}_{tgt}{suffix}.png"

    fig.savefig(fig_path, dpi=300)
    print(f"✅ Plot saved: {fig_path}")
    plt.close()


def main():
    """Run distributed communication benchmark."""
    # ========================================
    # PROFILING CONFIGURATION (modify here)
    # ========================================
    ENABLE_PROFILING = False  # Set to True to enable profiling
    PROFILE_SHAPES = [(8192, 8192)]  # Shapes to profile
    PROFILE_WARMUP = 2  # Warmup steps for profiling
    PROFILE_ACTIVE = 3  # Active profiling steps
    PROFILE_FREQ = 5  # Total frequency (warmup + active + wait)
    ENABLE_MEMORY_SNAPSHOT = False  # Generate memory snapshot pickle files

    # Distributed setup - torchrun will set these automatically
    device = "cuda"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup profiling configuration
    profiling_config = None
    if ENABLE_PROFILING:
        profiling_config = {
            "dump_folder": str(RESULTS_DIR),
            "profile_shapes": PROFILE_SHAPES,
            "warmup_steps": PROFILE_WARMUP,
            "active_steps": PROFILE_ACTIVE,
            "profile_freq": PROFILE_FREQ,
            "enable_memory_snapshot": ENABLE_MEMORY_SNAPSHOT,
        }

        if rank == 0:
            print(f"Profiling enabled for shapes: {PROFILE_SHAPES}")
            print(f"Memory snapshot: {ENABLE_MEMORY_SNAPSHOT}")

    # Initialize process group
    dist.init_process_group(backend="nccl" if device == "cuda" else "gloo")
    if device == "cuda":
        torch.cuda.set_device(local_rank)

    # Enable memory recording if memory snapshot is requested
    if ENABLE_PROFILING and ENABLE_MEMORY_SNAPSHOT:
        if device == "cuda":
            try:
                from utils import MEMORY_SNAPSHOT_MAX_ENTRIES

                torch.cuda.memory._record_memory_history(max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES)
                if rank == 0:
                    print(f"Memory recording enabled with max_entries={MEMORY_SNAPSHOT_MAX_ENTRIES}")
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Failed to enable memory recording: {e}")

    # SMOKE_TEST: shrink the sweep to a single-node 8-GPU run for quick
    # verification of the Partial path. Toggle off for the full benchmark.
    SMOKE_TEST = os.environ.get("ETHA_BENCH_SMOKE", "0") == "1"

    warmup_iter = 1 if SMOKE_TEST else 3
    profile_iter = 3 if SMOKE_TEST else 20
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 8))

    if SMOKE_TEST:
        # 4 source + 4 target = 8 GPUs; Partial on mesh dim 1 (size 2) is non-trivial.
        tensor_shapes = [(2048, 2048)]
        num_tensors_per_batch = 2
        mesh_combinations = [
            ((1, 2, 1, 2), (1, 1, 2, 2)),
        ]
    else:
        tensor_shapes = [
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
            (8192, 8192),
            (12288, 12288),
            (16384, 16384),
            (20480, 20480),
            (24576, 24576),
        ]
        num_tensors_per_batch = 25
        # 8 source + 8 target = 16 GPUs total
        mesh_combinations = [
            ((1, 1, 4, 2), (4, 2, 1, 1)),
            ((8, 1, 1, 1), (1, 2, 2, 2)),
            ((1, 2, 4, 1), (8, 1, 1, 1)),
            ((2, 2, 2, 1), (1, 1, 4, 2)),
            ((2, 4, 1, 1), (1, 2, 4, 1)),
            ((4, 1, 1, 2), (2, 1, 1, 4)),
            ((4, 1, 2, 1), (1, 1, 1, 8)),
            ((4, 2, 1, 1), (1, 1, 2, 4)),
        ]

    print(f"Total {len(mesh_combinations)} mesh combinations (smoke={SMOKE_TEST})")

    BENCH_CONFIGS: list[tuple[str, list]] = [
        ("no_partial", [Replicate(), Shard(0), Replicate(), Shard(dim=1)]),
        ("replicate_dp", [Replicate(), Replicate(), Replicate(), Shard(dim=1)]),
        ("partial_dp", [Replicate(), Partial("sum"), Replicate(), Shard(dim=1)]),
    ]

    print("Starting batch testing for all mesh combinations...")
    print(f"Total {len(mesh_combinations)} mesh combinations × {len(BENCH_CONFIGS)} configs to test")

    all_results = {}

    for config_name, source_specs in BENCH_CONFIGS:
        has_partial = any(isinstance(p, Partial) for p in source_specs)
        print(f"\n{'#' * 80}")
        print(f"# Config: {config_name} (Partial source: {has_partial})")
        print(f"# source_specs = {_placements_to_str(source_specs)}")
        print(f"{'#' * 80}")

        for mesh_idx, (source_mesh_shape, target_mesh_shape) in enumerate(mesh_combinations):
            # Create target_specs dynamically based on target_mesh_shape
            # split_factor should equal the size of mesh dimension 2 (the Shard(0) dimension)
            split_factor = target_mesh_shape[3]
            target_specs = [Replicate(), Replicate(), _StridedShard(0, split_factor=split_factor), Shard(0)]
            print(f"\n{'=' * 80}")
            print(f"Testing mesh combination {mesh_idx + 1}/{len(mesh_combinations)}:")
            print(f"Source mesh: {source_mesh_shape} -> Target mesh: {target_mesh_shape}")
            print(f"{'=' * 80}")

            # Setup current mesh
            source_world_size = math.prod(source_mesh_shape)
            target_world_size = math.prod(target_mesh_shape)

            print(
                f"Source mesh size: {source_world_size} devices, "
                f"Target mesh size: {target_world_size} devices, "
                f"Total: {source_world_size + target_world_size} devices"
            )

            # Create mesh
            current_source_mesh = DeviceMesh(device, torch.arange(source_world_size).view(source_mesh_shape))
            current_target_mesh = DeviceMesh(
                device, torch.arange(source_world_size, source_world_size + target_world_size).view(target_mesh_shape)
            )

            source_ranks_list = list(range(source_world_size))
            source_partial_groups = _build_source_partial_groups(
                current_source_mesh, source_specs, rank, source_ranks_list
            )

            # Run benchmark for current mesh
            current_results = []
            if device == "cuda":
                torch.cuda.synchronize()
            dist.barrier()
            start_time = time.perf_counter()
            m2m = get_m2m_map(
                source_mesh=current_source_mesh,
                source_placements=source_specs,
                target_mesh=current_target_mesh,
                target_placements=target_specs,
                group=dist.group.WORLD,
                device=device,
            )
            map_time = (time.perf_counter() - start_time) / profile_iter
            if rank == 0:
                print(f"get_m2m_map time: {map_time}")
            for shape in tensor_shapes:
                print(f"  Benchmarking batch of {num_tensors_per_batch} tensors with shape: {shape}...")

                is_in_source = rank < source_world_size

                # Only origin_tensors[0] is needed downstream (baseline reference);
                # keeping all 25 full-shape tensors on every rank blows past 100+ GiB
                # for shape=(24576, 24576).
                origin_tensors = []
                source_dist_tensors = []
                target_local_tensors = []

                for i in range(num_tensors_per_batch):
                    torch.manual_seed(i)  # Different seed for each tensor
                    origin_tensor = torch.randn(shape, device=device)

                    if is_in_source:
                        source_dist_tensor = distribute_tensor(origin_tensor, current_source_mesh, source_specs)
                        source_dist_tensors.append(source_dist_tensor)
                    else:
                        target_dist_tensor = distribute_tensor(origin_tensor, current_target_mesh, target_specs)
                        target_local_tensors.append(target_dist_tensor.to_local())

                    if i == 0:
                        origin_tensors.append(origin_tensor)
                    # else: drop the full reference so PyTorch can free it.

                # Generate chunks for all tensors and extend into one list
                if device == "cuda":
                    torch.cuda.synchronize()
                dist.barrier()

                start_time = time.perf_counter()

                all_chunks = []
                for i in range(num_tensors_per_batch):
                    if is_in_source:
                        source_local_tensor = source_dist_tensors[i].to_local()
                        target_local_tensor = None
                    else:
                        source_local_tensor = None
                        target_local_tensor = target_local_tensors[i]

                    chunks = m2m_to_chunks(
                        m2m,
                        rank=rank,
                        source_tensor=source_local_tensor,
                        target_tensor=target_local_tensor,
                        source_partial_groups=source_partial_groups,
                    )
                    all_chunks.extend(chunks)

                all_buckets = chunk_to_bucket_ops(
                    chunks=all_chunks,
                    bucket_size=BUCKET_SIZE_BYTES,
                )

                ir_gen_time = (time.perf_counter() - start_time) / profile_iter
                if rank == 0:
                    print(f"    IR generation time: {ir_gen_time:.6f}s")
                    print(
                        f"    Generated {len(all_chunks)} chunks total ({len(all_chunks) // num_tensors_per_batch} per tensor)"
                    )
                    print(
                        f"    Generated {len(all_buckets)} buckets total ({len(all_buckets) // num_tensors_per_batch} per tensor)"
                    )
                    print(f"    routes: {m2m.routes}")

                # Create mesh info string for output directory organization
                src = "_".join(map(str, source_mesh_shape))
                tgt = "_".join(map(str, target_mesh_shape))
                mesh_info = f"mesh_{mesh_idx + 1:02d}_{src}_{tgt}"

                result = benchmark_single_shape(
                    origin_tensors,
                    source_dist_tensors,
                    target_local_tensors,
                    num_tensors_per_batch,
                    shape,
                    all_chunks,
                    all_buckets,
                    current_source_mesh,
                    current_target_mesh,
                    source_specs,
                    target_specs,
                    source_world_size,
                    device,
                    rank,
                    local_rank,
                    warmup_iter,
                    profile_iter,
                    gpus_per_node,
                    profiling_config,
                    mesh_info,
                    has_partial=has_partial,
                )

                if result is not None:
                    current_results.append(result)

                # Clean up tensors after each shape benchmark
                del origin_tensors, source_dist_tensors, target_local_tensors, all_chunks, all_buckets
                if device == "cuda":
                    torch.cuda.empty_cache()

            # Generate plot for current mesh
            if rank == 0 and current_results:
                generate_result_plot(
                    current_results,
                    source_mesh_shape,
                    target_mesh_shape,
                    mesh_idx,
                    num_tensors_per_batch,
                    source_specs,
                    target_specs,
                    config_name=config_name,
                    has_partial=has_partial,
                )
                all_results[f"{config_name}_mesh_{mesh_idx + 1}_{source_mesh_shape}_{target_mesh_shape}"] = (
                    current_results
                )
                print(f"✅ [{config_name}] Mesh combination {mesh_idx + 1} test completed")

            # Clear process group cache and GPU memory after each mesh combination
            try:
                from etha.comm.comm_methods import _PROCESS_GROUP_CACHE

                _PROCESS_GROUP_CACHE.clear()
            except ImportError:
                pass
            if device == "cuda":
                torch.cuda.empty_cache()

            print(f"Mesh combination {mesh_idx + 1}/{len(mesh_combinations)} processing completed")

    if rank == 0:
        print(f"\n{'=' * 80}")
        print("All mesh combination tests completed!")
        print(f"Tested {len(mesh_combinations)} mesh combinations")
        print(f"Generated {len(all_results)} performance charts")
        print(f"All plots saved in: {RESULTS_DIR}")
        print(f"{'=' * 80}")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
