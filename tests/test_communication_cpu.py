"""Test communication methods on CPU."""

import os
import math
import logging

import torch
import pytest
import torch.distributed as dist
from torch.distributed._tensor import Shard, Replicate, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from etha.comm import (
    chunk_comm,
    bucket_comm,
    get_m2m_map,
    map_to_chunk_ops,
    chunk_to_bucket_ops,
    gather_broadcast_comm,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def run_test_communication(
    rank: int,
    world_size: int,
    source_mesh_shape: tuple[int, ...],
    target_mesh_shape: tuple[int, ...],
    device: str,
):
    dist.init_process_group(backend="nccl" if device == "cuda" else "gloo", rank=rank, world_size=world_size)
    source_world_size = math.prod(source_mesh_shape)
    target_world_size = math.prod(target_mesh_shape)

    source_mesh = DeviceMesh(device, torch.arange(source_world_size).view(source_mesh_shape))
    target_mesh = DeviceMesh(
        device,
        torch.arange(source_world_size, source_world_size + target_world_size).view(target_mesh_shape),
    )

    source_specs = [Replicate(), Replicate(), Shard(0), Shard(1)]
    target_specs = [Replicate(), _StridedShard(1, split_factor=2), Replicate(), Shard(1)]
    # Dummy tensor shape
    torch.manual_seed(0)
    shape = (64, 64)
    source_origin_tensor = torch.randn(shape, device=device)
    target_origin_tensor_chunk = torch.randn(shape, device=device)
    target_origin_tensor_bucket = torch.randn(shape, device=device)
    is_in_source = rank < source_world_size

    source_dist_tensor = distribute_tensor(source_origin_tensor, source_mesh, source_specs)
    source_local_tensor = source_dist_tensor.to_local()

    target_dist_tensor_chunk = distribute_tensor(target_origin_tensor_chunk, target_mesh, target_specs)
    target_local_chunk = target_dist_tensor_chunk.to_local()

    logger.debug(f"[rank={rank}] Source mesh: {source_mesh.mesh}")
    logger.debug(f"[rank={rank}] Target mesh: {target_mesh.mesh}")

    # Generate chunk IR using new API
    # Step 1: Get M2M map
    m2m_map, source_num_slicers, target_num_slicers = get_m2m_map(
        source_mesh=source_mesh,
        source_placements=source_specs,
        target_mesh=target_mesh,
        target_placements=target_specs,
        group=dist.group.WORLD,
        device=device,
    )
    if rank == 0:
        logger.info(f"Generated m2m map: {m2m_map}")
    # Step 2: Generate execution-ready chunks directly
    chunks = map_to_chunk_ops(
        m2m_map=m2m_map,
        rank=rank,
        source_num_slicers=source_num_slicers,
        target_num_slicers=target_num_slicers,
        source_tensor=source_local_tensor,
        target_tensor=target_local_chunk,
    )

    if rank == 0:
        logger.info(f"Generated {len(chunks)} chunks")

    # Test M2M communication with unified chunks
    chunk_comm(chunks=chunks)

    target_dist_tensor_bucket = distribute_tensor(target_origin_tensor_bucket, target_mesh, target_specs)
    target_local_bucket = target_dist_tensor_bucket.to_local()
    bucket_chunks = map_to_chunk_ops(
        m2m_map=m2m_map,
        rank=rank,
        source_num_slicers=source_num_slicers,
        target_num_slicers=target_num_slicers,
        source_tensor=source_local_tensor,
        target_tensor=target_local_bucket,
    )
    buckets = chunk_to_bucket_ops(
        chunks=bucket_chunks,
        bucket_size=256 * 1024,
    )
    logger.debug(f"[rank={rank}] Generated {len(buckets)} buckets")
    bucket_comm(buckets=buckets)
    logger.debug(f"[rank={rank}] Bucket communication completed")
    # Test Gather-Broadcast Method
    gather_broadcast_result = gather_broadcast_comm(
        target_mesh,
        target_specs,
        source_dist_tensor,
        target_origin_tensor_chunk,
        source_world_size,
    )
    if not is_in_source:
        chunk_result = target_dist_tensor_chunk.full_tensor()
        bucket_result = target_dist_tensor_bucket.full_tensor()
        assert torch.allclose(chunk_result, source_origin_tensor)
        assert torch.allclose(bucket_result, source_origin_tensor)
        if gather_broadcast_result is None:
            raise RuntimeError("Gather-Broadcast result missing on target rank.")
        assert torch.allclose(target_local_chunk, gather_broadcast_result.to_local())

    dist.destroy_process_group()


@pytest.mark.parametrize(
    "source_mesh_shape, target_mesh_shape",
    [
        ((2, 2, 1, 1), (2, 1, 1, 2)),
        ((2, 2, 1, 1), (2, 1, 2, 1)),
        ((2, 1, 2, 1), (2, 1, 1, 2)),
        ((2, 1, 1, 2), (1, 1, 2, 2)),
        ((1, 2, 2, 1), (1, 2, 1, 2)),
        ((1, 2, 1, 2), (1, 1, 2, 2)),
        ((1, 1, 2, 2), (2, 1, 1, 2)),
        ((4, 1, 1, 1), (1, 1, 1, 4)),
        ((1, 4, 1, 1), (1, 1, 4, 1)),
        ((1, 1, 4, 1), (1, 1, 1, 4)),
    ],
)
def test_communication_cpu(source_mesh_shape: tuple, target_mesh_shape: tuple):
    source_world_size = math.prod(source_mesh_shape)
    target_world_size = math.prod(target_mesh_shape)
    world_size = source_world_size + target_world_size
    device = "cpu"

    os.environ["MASTER_ADDR"] = "localhost"

    # Find an available port dynamically
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]

    os.environ["MASTER_PORT"] = str(port)

    # Use torch.multiprocessing.spawn to run the test in multiple processes
    # This is a common pattern for testing distributed PyTorch applications
    try:
        torch.multiprocessing.spawn(
            run_test_communication,
            args=(world_size, source_mesh_shape, target_mesh_shape, device),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        pytest.fail(f"Distributed test failed: {e}")
