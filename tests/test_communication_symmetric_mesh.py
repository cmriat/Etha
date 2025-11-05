"""Test communication methods on CPU."""

import os
import math
import logging

import torch
import pytest
import torch.distributed as dist
from torch.distributed._tensor import Shard, DeviceMesh, distribute_tensor

from etha.comm import (
    get_m2m_map,
    m2m_communicate,
    map_to_chunk_ops,
    gather_broadcast_communicate,
)

logging.basicConfig(level=logging.INFO)
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

    if rank < source_world_size:
        source_mesh = DeviceMesh(device, torch.arange(source_world_size).view(source_mesh_shape))
        target_mesh = DeviceMesh(
            device, torch.arange(source_world_size, source_world_size + target_world_size).view(target_mesh_shape)
        )
    else:
        target_mesh = DeviceMesh(device, torch.arange(source_world_size).view(source_mesh_shape))
        source_mesh = DeviceMesh(
            device, torch.arange(source_world_size, source_world_size + target_world_size).view(target_mesh_shape)
        )

    source_specs = [Shard(1)]
    target_specs = [Shard(1)]
    # Dummy tensor shape
    torch.manual_seed(0)
    shape = (64, 64)
    source_origin_tensor = torch.randn(shape, device=device)
    target_origin_tensor = torch.randn(shape, device=device)
    is_in_source = rank < source_world_size

    source_dist_tensor = None
    source_local_tensor = None
    if is_in_source:
        source_dist_tensor = distribute_tensor(source_origin_tensor, source_mesh, source_specs)
        source_local_tensor = source_dist_tensor.to_local()

    target_dist_tensor = None
    target_local_tensor = None
    if not is_in_source:
        target_dist_tensor = distribute_tensor(target_origin_tensor, source_mesh, target_specs)
        target_local_tensor = target_dist_tensor.to_local()

    logger.debug(f"[rank={rank}] Source mesh: {source_mesh.mesh}")
    logger.debug(f"[rank={rank}] Target mesh: {target_mesh.mesh}")

    # Generate chunk IR using new API
    # Step 1: Get M2M map
    if is_in_source:
        m2m_map, source_num_slicers, target_num_slicers = get_m2m_map(
            source_mesh=source_mesh,
            source_placements=source_specs,
            target_mesh=target_mesh,
            target_placements=target_specs,
            group=dist.group.WORLD,
            device=device,
        )
    else:
        m2m_map, source_num_slicers, target_num_slicers = get_m2m_map(
            source_mesh=target_mesh,
            source_placements=target_specs,
            target_mesh=source_mesh,
            target_placements=source_specs,
            group=dist.group.WORLD,
            device=device,
        )

    # Step 2: Generate execution-ready chunks directly
    chunks = map_to_chunk_ops(
        m2m_map=m2m_map,
        rank=rank,
        source_num_slicers=source_num_slicers,
        target_num_slicers=target_num_slicers,
        source_tensor=source_local_tensor,
        target_tensor=target_local_tensor,
    )

    if rank == 0:
        logger.info(f"Generated {len(chunks)} chunks")

    # Test M2M communication with unified chunks
    m2m_communicate(chunks=chunks)

    # Test Gather-Broadcast Method
    gather_broadcast_result = gather_broadcast_communicate(
        source_mesh,
        source_specs,
        source_dist_tensor,
        target_origin_tensor,
        source_world_size,
    )
    if not is_in_source:
        full_m2m_result = target_dist_tensor.full_tensor()

        assert torch.allclose(full_m2m_result, source_origin_tensor)
        # Assert results are close (due to potential floating point differences in distributed ops)
        if target_local_tensor is not None and gather_broadcast_result is not None:
            assert torch.allclose(target_local_tensor, gather_broadcast_result.to_local())
        else:
            raise RuntimeError(
                f"One result is None, the other is not. M2M: {target_local_tensor}, Gather-Broadcast: {gather_broadcast_result}"
            )

    dist.destroy_process_group()


@pytest.mark.parametrize(
    "source_mesh_shape, target_mesh_shape",
    [
        ((4,), (4,)),
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
