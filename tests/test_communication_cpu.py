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
    m2m_communicate,
    get_m2m_transfers,
    transfers_to_chunks,
    bind_tensors_to_chunks,
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
    target_origin_tensor = torch.randn(shape, device=device)
    is_in_source = rank < source_world_size

    source_dist_tensor = distribute_tensor(source_origin_tensor, source_mesh, source_specs)
    source_local_tensor = source_dist_tensor.to_local()

    target_dist_tensor = distribute_tensor(target_origin_tensor, target_mesh, target_specs)
    target_local_tensor = target_dist_tensor.to_local()

    logger.debug(f"[rank={rank}] Source mesh: {source_mesh.mesh}")
    logger.debug(f"[rank={rank}] Target mesh: {target_mesh.mesh}")

    # Generate chunk IR using new API
    # Step 1: Get Transfer IR
    transfers = get_m2m_transfers(
        source_mesh=source_mesh,
        source_placements=source_specs,
        target_mesh=target_mesh,
        target_placements=target_specs,
        group=dist.group.WORLD,
        device=device,
    )

    source_tensor_shape = None
    target_tensor_shape = None
    if source_local_tensor is not None and 0 not in source_local_tensor.shape:
        source_tensor_shape = tuple(source_local_tensor.shape)
    if target_local_tensor is not None and 0 not in target_local_tensor.shape:
        target_tensor_shape = tuple(target_local_tensor.shape)

    # Step 2: Generate chunk IR from Transfers
    source_chunks, target_chunks = transfers_to_chunks(
        transfers=transfers,
        rank=rank,
        source_tensor_shape=source_tensor_shape,
        target_tensor_shape=target_tensor_shape,
    )

    if rank == 0:
        logger.info(f"Generated {len(source_chunks)} source chunks, {len(target_chunks)} target chunks")

    # Bind tensors to chunks
    bind_tensors_to_chunks(source_chunks, target_chunks, source_local_tensor, target_local_tensor)

    # Test M2M communication with IR
    m2m_communicate(source_chunks, target_chunks)

    # Test Gather-Broadcast Method
    gather_broadcast_result = gather_broadcast_communicate(
        target_mesh,
        target_specs,
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
    os.environ["MASTER_PORT"] = "29500"

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
