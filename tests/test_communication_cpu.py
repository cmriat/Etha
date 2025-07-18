import math
import os
from typing import Tuple

import pytest
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, Replicate, Shard, distribute_tensor, DTensor
from torch.distributed.tensor.placement_types import _StridedShard

from rl_comm.communication_utils import (
    gather_broadcast_communicate,
    get_p2p_map,
    get_shard_tensor_shape,
    p2p_communicate,
)


def run_test_communication(
    rank: int,
    world_size: int,
    source_mesh_shape: Tuple[int, ...],
    target_mesh_shape: Tuple[int, ...],
    device: str,
):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    source_world_size = math.prod(source_mesh_shape)
    target_world_size = math.prod(target_mesh_shape)

    source_mesh = DeviceMesh(device, torch.arange(source_world_size).view(source_mesh_shape))
    target_mesh = DeviceMesh(
        device,
        torch.arange(source_world_size, source_world_size + target_world_size).view(
            target_mesh_shape
        ),
    )

    source_specs = [Replicate(), Replicate(), Shard(0), Shard(1)]
    target_specs = [Replicate(), _StridedShard(1, split_factor=2), Replicate(), Shard(1)]
    # Dummy tensor shape
    torch.manual_seed(0)
    shape = (64, 64)
    origin_tensor = torch.randn(shape, device=device)

    is_in_source = rank < source_world_size
    local_tensor = None
    source_dist_tensor = None
    if is_in_source:
        source_dist_tensor = distribute_tensor(origin_tensor, source_mesh, source_specs)
        local_tensor = source_dist_tensor.to_local()

    target_local_shape = get_shard_tensor_shape(
        origin_tensor.shape, target_mesh, target_specs
    )

    # Test P2P Map Method
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
    if rank == 0:
        print(f"Forward Map: {forward_map}")
        print(f"Reverse Map: {reverse_map}")
        print(f"Source Num Slicers: {source_num_slicers}")
        print(f"Target Num Slicers: {target_num_slicers}")
    p2p_result = p2p_communicate(
        rank,
        forward_map,
        reverse_map,
        local_tensor,
        source_num_slicers,
        target_num_slicers,
        target_local_shape,
        device,
    )

    # Test Gather-Broadcast Method
    gather_broadcast_result = gather_broadcast_communicate(
        rank,
        source_mesh,
        source_specs,
        target_mesh,
        target_specs,
        source_dist_tensor,
        origin_tensor,
        source_world_size,
        device,
    )
    if not is_in_source:
        p2p_result_local = DTensor.from_local(p2p_result, target_mesh, target_specs)
        full_p2p_result = p2p_result_local.full_tensor()
        assert torch.allclose(full_p2p_result, origin_tensor)
        # Assert results are close (due to potential floating point differences in distributed ops)
        if p2p_result is not None and gather_broadcast_result is not None:
            assert torch.allclose(p2p_result, gather_broadcast_result.to_local())
        elif p2p_result is None and gather_broadcast_result is None:
            pass  # Both are None, which is expected if rank is not in target mesh
        else:
            assert False, f"One result is None, the other is not. P2P: {p2p_result}, Gather-Broadcast: {gather_broadcast_result}"

    dist.destroy_process_group()

@pytest.mark.parametrize(
    "source_mesh_shape, target_mesh_shape",
    [
        ((2, 2, 2, 2), (2, 2, 2, 2)),
        ((2, 2, 2, 2), (2, 2, 2, 4)),
        ((2, 2, 2, 2), (2, 2, 4, 2)),
        ((2, 2, 2, 2), (2, 2, 4, 4)),
        ((2, 2, 2, 2), (2, 4, 2, 2)),
        ((2, 2, 2, 2), (2, 4, 2, 4)),
        ((2, 2, 2, 2), (2, 4, 4, 2)),
        ((2, 2, 2, 2), (2, 4, 4, 4)),
        ((2, 2, 2, 2), (4, 2, 2, 2)),
        ((2, 2, 2, 2), (4, 2, 2, 4)),
        ((2, 2, 2, 4), (2, 2, 2, 2)),
        ((2, 2, 2, 4), (2, 2, 2, 4)),
        ((2, 2, 2, 4), (2, 2, 4, 2)),
        ((2, 2, 2, 4), (2, 2, 4, 4)),
        ((2, 2, 2, 4), (2, 4, 2, 2)),
        ((2, 2, 2, 4), (2, 4, 2, 4)),
        ((2, 2, 2, 4), (2, 4, 4, 2)),
        ((2, 2, 2, 4), (2, 4, 4, 4)),
        ((2, 2, 2, 4), (4, 2, 2, 2)),
        ((2, 2, 2, 4), (4, 2, 2, 4)),
        ((2, 2, 4, 2), (2, 2, 2, 2)),
        ((2, 2, 4, 2), (2, 2, 2, 4)),
        ((2, 2, 4, 2), (2, 2, 4, 2)),
        ((2, 2, 4, 2), (2, 2, 4, 4)),
        ((2, 2, 4, 2), (2, 4, 2, 2)),
        ((2, 2, 4, 2), (2, 4, 2, 4)),
        ((2, 2, 4, 2), (2, 4, 4, 2)),
        ((2, 2, 4, 2), (2, 4, 4, 4)),
        ((2, 2, 4, 2), (4, 2, 2, 2)),
        ((2, 2, 4, 2), (4, 2, 2, 4)),
        ((2, 2, 4, 4), (2, 2, 2, 2)),
        ((2, 2, 4, 4), (2, 2, 2, 4)),
        ((2, 2, 4, 4), (2, 2, 4, 2)),
        ((2, 2, 4, 4), (2, 2, 4, 4)),
        ((2, 2, 4, 4), (2, 4, 2, 2)),
        ((2, 2, 4, 4), (2, 4, 2, 4)),
        ((2, 2, 4, 4), (2, 4, 4, 2)),
        ((2, 2, 4, 4), (2, 4, 4, 4)),
        ((2, 2, 4, 4), (4, 2, 2, 2)),
        ((2, 2, 4, 4), (4, 2, 2, 4)),
        ((2, 4, 2, 2), (2, 2, 2, 2)),
        ((2, 4, 2, 2), (2, 2, 2, 4)),
        ((2, 4, 2, 2), (2, 2, 4, 2)),
        ((2, 4, 2, 2), (2, 2, 4, 4)),
        ((2, 4, 2, 2), (2, 4, 2, 2)),
        ((2, 4, 2, 2), (2, 4, 2, 4)),
        ((2, 4, 2, 2), (2, 4, 4, 2)),
        ((2, 4, 2, 2), (2, 4, 4, 4)),
        ((2, 4, 2, 2), (4, 2, 2, 2)),
        ((2, 4, 2, 2), (4, 2, 2, 4)),
    ],
)
def test_communication_cpu(source_mesh_shape: Tuple, target_mesh_shape: Tuple):
    source_world_size = math.prod(source_mesh_shape)
    target_world_size = math.prod(target_mesh_shape)
    world_size = source_world_size + target_world_size
    device = "cpu"

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

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
