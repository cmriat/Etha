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

from rockstar import (
    get_p2p_map,
    p2p_communicate,
    get_shard_tensor_shape,
    gather_broadcast_communicate,
)


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

    # 16卡的mesh组合 (source + target = 16, 每个维度是2的幂次: 1,2,4,8)
    # 混合不同的source和target组合
    mesh_combinations = [
        # 8+8=16的不同组合
        # ((1, 1, 1, 8), (2, 2, 2, 1)),  # 线性 vs 紧凑
        # ((1, 1, 2, 4), (1, 2, 2, 2)),  # 中等 vs 平衡
        # ((1, 1, 4, 2), (4, 2, 1, 1)),  # 宽 vs 高
        # ((1, 2, 1, 4), (1, 4, 2, 1)),  # 不同维度分布
        # ((1, 2, 2, 2), (2, 1, 4, 1)),  # 平衡 vs 长条
        # ((1, 2, 4, 1), (8, 1, 1, 1)),  # 扁平 vs 线性
        # ((1, 4, 1, 2), (1, 1, 8, 1)),  # 中等 vs 单维
        # ((1, 4, 2, 1), (1, 8, 1, 1)),  # 宽 vs 深
        # ((1, 8, 1, 1), (2, 2, 1, 2)),  # 单维 vs 分布
        # ((2, 1, 1, 4), (1, 1, 2, 4)),  # 长 vs 宽
        # ((2, 1, 2, 2), (1, 2, 1, 4)),  # 方形 vs 线性
        # ((2, 1, 4, 1), (4, 1, 2, 1)),  # 条状 vs 条状
        # ((2, 2, 1, 2), (1, 4, 1, 2)),  # 原始 vs 变形
        ((2, 2, 2, 1), (1, 1, 4, 2)),  # 立方 vs 扁平
        ((2, 4, 1, 1), (1, 2, 4, 1)),  # 矩形不同方向
        ((4, 1, 1, 2), (2, 1, 1, 4)),  # 对称变换
        ((4, 1, 2, 1), (1, 1, 1, 8)),  # 分布 vs 线性
        ((4, 2, 1, 1), (1, 1, 2, 4)),  # 紧凑 vs 分布
        ((8, 1, 1, 1), (1, 2, 2, 2)),  # 线性 vs 立方
    ]

    print(f"总共有 {len(mesh_combinations)} 个不同的16卡mesh组合")

    source_specs = [Replicate(), Shard(0), Replicate(), Shard(1)]
    target_specs = [Replicate(), Replicate(), Shard(0), Shard(1)]

    # 为所有mesh组合运行benchmark并生成图片
    print("开始批量测试所有mesh组合...")
    print(f"总共有 {len(mesh_combinations)} 个mesh组合需要测试")

    all_results = {}  # 存储所有mesh组合的结果

    for mesh_idx, (source_mesh_shape, target_mesh_shape) in enumerate(mesh_combinations):
        print(f"\n{'=' * 80}")
        print(f"测试第 {mesh_idx + 1}/{len(mesh_combinations)} 个mesh组合:")
        print(f"源mesh: {source_mesh_shape} -> 目标mesh: {target_mesh_shape}")
        print(f"{'=' * 80}")

        # 设置当前mesh
        source_world_size = math.prod(source_mesh_shape)
        target_world_size = math.prod(target_mesh_shape)

        print(
            f"源mesh大小: {source_world_size}卡, 目标mesh大小: {target_world_size}卡, 总计: {source_world_size + target_world_size}卡"
        )

        # 创建mesh
        current_source_mesh = DeviceMesh(device, torch.arange(source_world_size).view(source_mesh_shape))
        current_target_mesh = DeviceMesh(
            device, torch.arange(source_world_size, source_world_size + target_world_size).view(target_mesh_shape)
        )

        # 为当前mesh运行benchmark
        current_results = []
        if device == "cuda":
            torch.cuda.synchronize(device=f"cuda:{local_rank}")
        dist.barrier(device_ids=[local_rank])

        start_time = time.perf_counter()
        forward_map, reverse_map, source_num_slicers, target_num_slicers = get_p2p_map(
            current_source_mesh,
            source_specs,
            current_target_mesh,
            target_specs,
            device,
        )
        map_time = (time.perf_counter() - start_time) / profile_iter
        print(f"get_p2p_map time: {map_time}")
        print(f"Forward map: {forward_map}")
        print(f"Reverse map: {reverse_map}")

        for shape in tensor_shapes:
            print(f"  Benchmarking tensor shape: {shape}...")
            torch.manual_seed(0)
            origin_tensor = torch.randn(shape, device=device)
            tensor_size_bytes = origin_tensor.nelement() * origin_tensor.element_size()
            tensor_size_gb = tensor_size_bytes / (1024**3)

            is_in_source = rank < source_world_size

            source_dist_tensor = None
            if is_in_source:
                source_dist_tensor = distribute_tensor(origin_tensor, current_source_mesh, source_specs)

            target_local_shape = get_shard_tensor_shape(origin_tensor.shape, current_target_mesh, target_specs)
            p2p_result = None
            # P2P方法benchmark
            dist.barrier(device_ids=[local_rank])
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
                torch.cuda.synchronize(device=f"cuda:{local_rank}")
            dist.barrier(device_ids=[local_rank])

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
                torch.cuda.synchronize(device=f"cuda:{local_rank}")
            dist.barrier(device_ids=[local_rank])
            p2p_time = (time.perf_counter() - start_time) / profile_iter

            bc_result = None
            # Gather-Broadcast方法benchmark
            dist.barrier(device_ids=[local_rank])
            for _ in range(warmup_iter):
                bc_result = gather_broadcast_communicate(
                    current_target_mesh,
                    target_specs,
                    source_dist_tensor,
                    origin_tensor,
                    source_world_size,
                    device,
                )
            if p2p_result is not None and bc_result is not None:
                assert torch.allclose(p2p_result, bc_result.to_local()), "P2P和Gather-Broadcast方法的结果不一致！"
            if device == "cuda":
                torch.cuda.synchronize(device=f"cuda:{local_rank}")
            dist.barrier(device_ids=[local_rank])

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
                torch.cuda.synchronize(device=f"cuda:{local_rank}")
            dist.barrier(device_ids=[local_rank])
            baseline_time = (time.perf_counter() - start_time) / profile_iter

            if rank == 0:
                p2p_throughput = tensor_size_gb / p2p_time
                baseline_throughput = tensor_size_gb / baseline_time
                current_results.append(
                    {
                        "tensor_shape": str(shape),
                        "tensor_size_mb": tensor_size_bytes / (1024**2),
                        "p2p_throughput_gb_s": p2p_throughput,
                        "baseline_throughput_gb_s": baseline_throughput,
                    }
                )

        # 为当前mesh生成图片
        if rank == 0 and current_results:
            df = pd.DataFrame(current_results)

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

            ax.set_title(
                f"P2P vs Gather-Broadcast Throughput\nMesh {mesh_idx + 1}/{len(mesh_combinations)}: {source_mesh_shape} -> {target_mesh_shape}",
                fontsize=16,
                weight="bold",
            )
            ax.set_xlabel("Tensor Size (MB)", fontsize=12)
            ax.set_ylabel("Throughput (GB/s)", fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)

            plt.tight_layout()

            # 生成文件名
            mesh_filename = (
                f"mesh_{mesh_idx + 1:02d}_{source_mesh_shape}_{target_mesh_shape}".replace(" ", "")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "_")
            )
            fig_path = f"./results/throughput_benchmark_{mesh_filename}.png"
            fig.savefig(fig_path, dpi=300)
            print(f"✅ 图片已保存: {fig_path}")

            plt.close()

            # 保存结果
            all_results[f"mesh_{mesh_idx + 1}_{source_mesh_shape}_{target_mesh_shape}"] = current_results

            print(f"✅ Mesh组合 {mesh_idx + 1} 测试完成")

        print(f"Mesh组合 {mesh_idx + 1}/{len(mesh_combinations)} 处理完成")

    if rank == 0:
        print(f"\n{'=' * 80}")
        print("所有mesh组合测试完成！")
        print(f"总共测试了 {len(mesh_combinations)} 个mesh组合")
        print(f"生成了 {len(all_results)} 个性能图表")
        print("所有图片保存在: ./results/")
        print(f"{'=' * 80}")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
