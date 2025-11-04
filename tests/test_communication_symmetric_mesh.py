"""Test communication methods on CPU."""

import os
import math
import logging

import torch
import pytest
import torch.distributed as dist
from torch.distributed._tensor import Shard, DeviceMesh, distribute_tensor

from etha.comm import (
    get_p2p_map,
    p2p_communicate,
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
    # Test P2P Map Method
    if is_in_source:
        forward_map, reverse_map, source_num_slicers, target_num_slicers = get_p2p_map(
            source_mesh,
            source_specs,
            target_mesh,
            target_specs,
            dist.group.WORLD,
            device,
        )
        forward_map_2, reverse_map_2, source_num_slicers_2, target_num_slicers_2 = get_p2p_map(
            target_mesh,
            target_specs,
            source_mesh,
            source_specs,
            dist.group.WORLD,
            device,
        )
    else:
        forward_map, reverse_map, source_num_slicers, target_num_slicers = get_p2p_map(
            target_mesh,
            target_specs,
            source_mesh,
            source_specs,
            dist.group.WORLD,
            device,
        )
        forward_map_2, reverse_map_2, source_num_slicers_2, target_num_slicers_2 = get_p2p_map(
            source_mesh,
            source_specs,
            target_mesh,
            target_specs,
            dist.group.WORLD,
            device,
        )
    if rank == 0:
        logger.info(f"Forward Map: {forward_map}")
        logger.info(f"Reverse Map: {reverse_map}")
        logger.info(f"Source Num Slicers: {source_num_slicers}")
        logger.info(f"Target Num Slicers: {target_num_slicers}")
        logger.info(f"Forward Map 2: {forward_map_2}")
        logger.info(f"Reverse Map 2: {reverse_map_2}")
        logger.info(f"Source Num Slicers 2: {source_num_slicers_2}")
        logger.info(f"Target Num Slicers 2: {target_num_slicers_2}")
    p2p_communicate(
        source_local_tensor,
        target_local_tensor,
        forward_map,
        reverse_map,
        source_num_slicers,
        target_num_slicers,
    )

    # Test Gather-Broadcast Method
    gather_broadcast_result = gather_broadcast_communicate(
        source_mesh,
        source_specs,
        source_dist_tensor,
        target_origin_tensor,
        source_world_size,
    )
    if not is_in_source:
        full_p2p_result = target_dist_tensor.full_tensor()

        assert torch.allclose(full_p2p_result, source_origin_tensor)
        # Assert results are close (due to potential floating point differences in distributed ops)
        if target_local_tensor is not None and gather_broadcast_result is not None:
            assert torch.allclose(target_local_tensor, gather_broadcast_result.to_local())
        else:
            raise RuntimeError(
                f"One result is None, the other is not. P2P: {target_local_tensor}, Gather-Broadcast: {gather_broadcast_result}"
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
