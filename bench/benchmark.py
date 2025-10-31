"""Distributed communication benchmark for P2P vs Gather-Broadcast methods."""

import os
import math
import time

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.distributed._tensor import Shard, Replicate, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from etha.comm import (
    get_p2p_map,
    p2p_communicate,
    get_shard_tensor_shape,
    gather_broadcast_communicate,
)

# Hardware bandwidth parameters (GB/s)
RDMA_SEND_BANDWIDTH_GB_S = 50.0
NVLINK_Single_BANDWIDTH_GB_S = 450.0


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
    shape: tuple,
    origin_tensor: torch.Tensor,
    forward_map,
    reverse_map,
    source_dist_tensor,
    source_num_slicers,
    target_num_slicers,
    target_local_shape,
    current_source_mesh: DeviceMesh,
    current_target_mesh: DeviceMesh,
    source_specs: list,  # noqa
    target_specs: list,
    source_world_size: int,
    device: str,
    rank: int,
    local_rank: int,  # noqa
    warmup_iter: int,
    profile_iter: int,
    gpus_per_node: int,
):
    """Benchmark P2P vs Gather-Broadcast for a single tensor shape.

    Returns:
        dict with benchmark results if rank == 0, None otherwise.
    """
    tensor_size_bytes = origin_tensor.nelement() * origin_tensor.element_size()

    # P2P method warmup
    dist.barrier()
    for _ in range(warmup_iter):
        p2p_result = p2p_communicate(
            forward_map,
            reverse_map,
            source_dist_tensor,
            source_num_slicers,
            target_num_slicers,
            target_local_shape,
            device=device,
            dtype=torch.float32,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    # P2P method benchmark
    start_time = time.perf_counter()
    for _ in range(profile_iter):
        p2p_communicate(
            forward_map,
            reverse_map,
            source_dist_tensor,
            source_num_slicers,
            target_num_slicers,
            target_local_shape,
            device=device,
            dtype=torch.float32,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    dist.barrier()
    p2p_time = (time.perf_counter() - start_time) / profile_iter

    # Gather-Broadcast method warmup
    dist.barrier()
    for _ in range(warmup_iter):
        bc_result = gather_broadcast_communicate(
            current_target_mesh,
            target_specs,
            source_dist_tensor,
            origin_tensor,
            source_world_size,
            device,
        )

    # Verify correctness
    if p2p_result is not None and bc_result is not None:
        if not torch.allclose(p2p_result, bc_result.to_local()):
            print(f"[Rank {rank}] P2P result shape: {p2p_result.shape}")
            print(f"[Rank {rank}] Baseline result shape: {bc_result.to_local().shape}")
            print(f"[Rank {rank}] Max diff: {(p2p_result - bc_result.to_local()).abs().max().item()}")
            print(f"[Rank {rank}] P2P sample: {p2p_result}")
            print(f"[Rank {rank}] Baseline sample: {bc_result.to_local()}")
            raise ValueError("P2P and Gather-Broadcast results mismatch!")

    if device == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    # Gather-Broadcast method benchmark
    start_time = time.perf_counter()
    for _ in range(profile_iter):
        gather_broadcast_communicate(
            current_target_mesh,
            target_specs,
            source_dist_tensor,
            origin_tensor,
            source_world_size,
            device,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    dist.barrier()
    baseline_time = (time.perf_counter() - start_time) / profile_iter

    if rank == 0:
        # Calculate ideal bytes that target ranks need to receive
        ideal_transfer_bytes = calculate_transfer_bytes(
            current_target_mesh, target_specs, origin_tensor.element_size() * origin_tensor.nelement()
        )
        ideal_transfer_gb = ideal_transfer_bytes / (1024**3)

        # Calculate ideal bandwidth (hardware topology aware)
        ideal_bw = calculate_ideal_bandwidth(
            current_source_mesh,
            current_target_mesh,
            target_specs,
            tensor_size_bytes,
            gpus_per_node,
        )

        # Effective throughput: how fast we complete the redistribution task
        p2p_effective_throughput = ideal_transfer_gb / p2p_time
        baseline_effective_throughput = ideal_transfer_gb / baseline_time

        print(f"    Origin tensor: {tensor_size_bytes / (1024**2):.2f} MB")
        print(f"    Ideal P2P transfer (target needs): {ideal_transfer_bytes / (1024**2):.2f} MB")
        print(f"    Ideal bandwidth (RDMA+NVLink): {ideal_bw:.2f} GB/s")
        print(f"    P2P effective throughput: {p2p_effective_throughput:.2f} GB/s")
        print(f"    Baseline effective throughput: {baseline_effective_throughput:.2f} GB/s")

        return {
            "tensor_shape": str(shape),
            "tensor_size_mb": tensor_size_bytes / (1024**2),
            "ideal_transfer_mb": ideal_transfer_bytes / (1024**2),
            "ideal_bandwidth_gb_s": ideal_bw,
            "p2p_effective_throughput_gb_s": p2p_effective_throughput,
            "baseline_effective_throughput_gb_s": baseline_effective_throughput,
        }

    return None


def generate_result_plot(
    results: list,
    source_mesh_shape: tuple,
    target_mesh_shape: tuple,
    mesh_idx: int,
):
    """Generate and save throughput comparison plot."""
    df = pd.DataFrame(results)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Plot all methods including ideal bandwidth
    plot_configs = [
        ("ideal_bandwidth_gb_s", "Ideal (RDMA+NVLink)", "--", "green", None),
        ("p2p_effective_throughput_gb_s", "P2P Effective", "o", "blue", 8),
        ("baseline_effective_throughput_gb_s", "Gather-Broadcast", "X", "orange", 8),
    ]

    for y_col, label, marker, color, markersize in plot_configs:
        if marker == "--":
            # Dashed line for ideal bandwidth
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

    ax.set_title(
        f"Effective Throughput (Task Speed)\nMesh: {source_mesh_shape} → {target_mesh_shape}",
        fontsize=14,
        weight="bold",
    )
    ax.set_xlabel("Tensor Size (MB)", fontsize=11)
    ax.set_ylabel("Effective Throughput (GB/s)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    # Generate clean filename
    src = "_".join(map(str, source_mesh_shape))
    tgt = "_".join(map(str, target_mesh_shape))
    fig_path = f"./results/throughput_benchmark_mesh_{mesh_idx + 1:02d}_{src}_{tgt}.png"

    fig.savefig(fig_path, dpi=300)
    print(f"✅ Plot saved: {fig_path}")
    plt.close()


def main():
    """Run distributed communication benchmark."""
    # Create results directory
    os.makedirs("./results", exist_ok=True)

    # Distributed setup - torchrun will set these automatically
    device = "cuda"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize process group
    if device == "cuda":
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:
        dist.init_process_group(backend="gloo")

    # BENCHMARKING PARAMETERS
    warmup_iter = 20
    profile_iter = 50
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 8))  # Default to 8 GPUs per node

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
        (28672, 28672),
        (32768, 32768),
    ]

    # Mesh combinations: source_mesh_shape -> target_mesh_shape (total 16 devices)
    # Each dimension must be power of 2: 1, 2, 4, 8
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

    print(f"Total {len(mesh_combinations)} different 16-device mesh combinations")

    source_specs = [Replicate(), Shard(0), Replicate(), Shard(dim=1)]

    # Run benchmark for all mesh combinations and generate plots
    print("Starting batch testing for all mesh combinations...")
    print(f"Total {len(mesh_combinations)} mesh combinations to test")

    all_results = {}  # Store results for all mesh combinations

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

        # Run benchmark for current mesh
        current_results = []
        if device == "cuda":
            torch.cuda.synchronize()
        dist.barrier()

        start_time = time.perf_counter()
        forward_map, reverse_map, source_num_slicers, target_num_slicers = get_p2p_map(
            current_source_mesh,
            source_specs,
            current_target_mesh,
            target_specs,
            device,
        )
        map_time = (time.perf_counter() - start_time) / profile_iter
        if rank == 0:
            print(f"get_p2p_map time: {map_time}")

        for shape in tensor_shapes:
            print(f"  Benchmarking tensor shape: {shape}...")
            torch.manual_seed(0)
            origin_tensor = torch.randn(shape, device=device)

            is_in_source = rank < source_world_size
            source_dist_tensor = None
            if is_in_source:
                source_dist_tensor = distribute_tensor(origin_tensor, current_source_mesh, source_specs)

            target_local_shape = get_shard_tensor_shape(shape, current_target_mesh, target_specs)

            result = benchmark_single_shape(
                shape,
                origin_tensor,
                forward_map,
                reverse_map,
                source_dist_tensor,
                source_num_slicers,
                target_num_slicers,
                target_local_shape,
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
            )

            if result is not None:
                current_results.append(result)

            # Clean up tensors after each shape benchmark
            del origin_tensor, source_dist_tensor
            if device == "cuda":
                torch.cuda.empty_cache()

        # Generate plot for current mesh
        if rank == 0 and current_results:
            generate_result_plot(current_results, source_mesh_shape, target_mesh_shape, mesh_idx)
            all_results[f"mesh_{mesh_idx + 1}_{source_mesh_shape}_{target_mesh_shape}"] = current_results
            print(f"✅ Mesh combination {mesh_idx + 1} test completed")

        # Clear process group cache and GPU memory after each mesh combination
        try:
            from etha.communication_utils import _PROCESS_GROUP_CACHE

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
        print("All plots saved in: ./results/")
        print(f"{'=' * 80}")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
