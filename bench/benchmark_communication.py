import math
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from rl_comm import (  # type: ignore
    gather_broadcast_communicate,
    get_p2p_map,
    get_shard_tensor_shape,
    p2p_communicate,
)


if __name__ == "__main__":
    device = "cuda"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)

    # BENCHMARKING PARAMETERS
    warmup_iter = 20
    profile_iter = 50
    tensor_shapes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (12288, 12888),
        (16384, 16384),
        (20480, 20480),
        (24576, 24576),
        (28672, 28672),
        (32768, 32768),
    ]

    source_mesh_shape = (2, 2, 1, 2)
    target_mesh_shape = (1, 1, 2, 4)
    source_world_size = math.prod(source_mesh_shape)
    target_world_size = math.prod(target_mesh_shape)

    source_mesh = DeviceMesh(device, torch.arange(source_world_size).view(source_mesh_shape))
    target_mesh = DeviceMesh(
        device, torch.arange(source_world_size, world_size).view(target_mesh_shape)
    )

    source_specs = [Replicate(), _StridedShard(1, split_factor=2), Replicate(), Shard(1)]
    target_specs = [Replicate(), _StridedShard(1, split_factor=2), Replicate(), Shard(1)]

    results = []

    for shape in tensor_shapes:
        if rank == 0:
            print(f"\nBenchmarking for tensor shape: {shape}...")

        forward_map, reverse_map, source_num_slicers, target_num_slicers = get_p2p_map(
            source_mesh,
            source_specs,
            target_mesh,
            target_specs,
            rank,
            source_world_size,
            target_world_size,
            device,
        )

        torch.manual_seed(0)
        origin_tensor = torch.randn(shape, device=device)
        tensor_size_bytes = origin_tensor.nelement() * origin_tensor.element_size()
        tensor_size_gb = tensor_size_bytes / (1024**3)

        is_in_source = rank < source_world_size

        local_tensor = None
        if is_in_source:
            source_dist_tensor = distribute_tensor(origin_tensor, source_mesh, source_specs)
            local_tensor = source_dist_tensor.to_local()

        target_local_shape = get_shard_tensor_shape(
            origin_tensor.shape, target_mesh, target_specs
        )

        # --- 1. Benchmark P2P Map Method ---
        dist.barrier(device_ids=[local_rank])
        for _ in range(warmup_iter):
            p2p_communicate(
                rank,
                forward_map,
                reverse_map,
                local_tensor,
                source_num_slicers,
                target_num_slicers,
                target_local_shape,
                device=device,
            )
        torch.cuda.synchronize(device=f"cuda:{local_rank}")
        dist.barrier(device_ids=[local_rank])

        start_time = time.perf_counter()
        for _ in range(profile_iter):
            p2p_communicate(
                rank,
                forward_map,
                reverse_map,
                local_tensor,
                source_num_slicers,
                target_num_slicers,
                target_local_shape,
                device=device,
            )
        torch.cuda.synchronize(device=f"cuda:{local_rank}")
        dist.barrier(device_ids=[local_rank])

        p2p_time = (time.perf_counter() - start_time) / profile_iter

        # --- 2. Benchmark Gather-Broadcast Method ---
        dist.barrier(device_ids=[local_rank])
        for _ in range(warmup_iter):
            gather_broadcast_communicate(
                rank,
                source_mesh,
                source_specs,
                target_mesh,
                target_specs,
                local_tensor,
                origin_tensor,
                source_world_size,
                device,
            )
        torch.cuda.synchronize(device=f"cuda:{local_rank}")
        dist.barrier(device_ids=[local_rank])
        start_time = time.perf_counter()
        for _ in range(profile_iter):
            gather_broadcast_communicate(
                rank,
                source_mesh,
                source_specs,
                target_mesh,
                target_specs,
                local_tensor,
                origin_tensor,
                source_world_size,
                device,
            )
        torch.cuda.synchronize(device=f"cuda:{local_rank}")
        dist.barrier(device_ids=[local_rank])

        baseline_time = (time.perf_counter() - start_time) / profile_iter

        if rank == 0:
            p2p_throughput = tensor_size_gb / p2p_time
            baseline_throughput = tensor_size_gb / baseline_time
            results.append(
                {
                    "tensor_shape": str(shape),
                    "tensor_size_mb": tensor_size_bytes / (1024**2),
                    "p2p_throughput_gb_s": p2p_throughput,
                    "baseline_throughput_gb_s": baseline_throughput,
                }
            )

    # --- Plot Results ---
    if rank == 0:
        df = pd.DataFrame(results)
        print("\n" + "=" * 50)
        print("--- Benchmark Results ---")
        print(df)
        print("=" * 50)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))

        sns.lineplot(
            data=df,
            x="tensor_size_mb",
            y="p2p_throughput_gb_s",
            marker="o",
            markersize=8,
            label="P2P Map Method",
            ax=ax,
        )
        sns.lineplot(
            data=df,
            x="tensor_size_mb",
            y="baseline_throughput_gb_s",
            marker="X",
            markersize=8,
            label="Gather-Broadcast Method",
            ax=ax,
        )

        ax.set_title("P2P vs Gather-Broadcast Throughput", fontsize=16, weight="bold")
        ax.set_xlabel("Tensor Size (MB)", fontsize=12)
        ax.set_ylabel("Throughput (GB/s)", fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Improve layout and save the figure
        plt.tight_layout()
        fig.savefig("throughput_benchmark.png", dpi=300)
        print("Benchmark plot saved to throughput_benchmark.png")

    else:
        print("Plotting libraries (matplotlib, seaborn) not found. Skipping plot generation.")

    dist.destroy_process_group()

