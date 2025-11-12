"""Test communication methods on CPU."""

import os
import math
import logging

import torch
import pytest
import torch.distributed as dist
from torch.distributed._tensor import Shard, DeviceMesh, distribute_tensor

from etha.comm import (
    chunk_comm,
    get_m2m_map,
    map_to_chunk_ops,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_test_communication(
    rank: int,
    world_size: int,
    mesh_shape: tuple[int, ...],
    device: str,
):
    """Test symmetric mesh communication where both groups send and receive.

    This test verifies that calling get_m2m_map twice with reversed source/target
    does not cause deadlocks. Both groups act as sender and receiver.
    """
    dist.init_process_group(backend="nccl" if device == "cuda" else "gloo", rank=rank, world_size=world_size)
    mesh_size = math.prod(mesh_shape)

    # All ranks use consistent mesh definitions
    mesh_a = DeviceMesh(device, torch.arange(mesh_size).view(mesh_shape))
    mesh_b = DeviceMesh(device, torch.arange(mesh_size, mesh_size * 2).view(mesh_shape))

    specs = [Shard(1)]

    # Dummy tensor shape
    torch.manual_seed(0)
    shape = (64, 64)
    tensor_a_origin = torch.randn(shape, device=device)
    tensor_b_origin = torch.randn(shape, device=device)

    is_in_mesh_a = rank < mesh_size

    # Each rank belongs to one mesh and creates a distributed tensor
    if is_in_mesh_a:
        dist_tensor_a = distribute_tensor(tensor_a_origin, mesh_a, specs)
        local_tensor = dist_tensor_a.to_local()
    else:
        # Create target distributed tensor with random data (will be overwritten)
        dist_tensor_b = distribute_tensor(tensor_b_origin, mesh_b, specs)
        local_tensor = dist_tensor_b.to_local()

    logger.debug(f"[rank={rank}] Mesh A: {mesh_a.mesh}, Mesh B: {mesh_b.mesh}")

    # Test: Call get_m2m_map twice with reversed source/target to test for deadlock
    # Direction 1: A -> B
    m2m_map_a_to_b, source_slicers_a, target_slicers_b = get_m2m_map(
        source_mesh=mesh_a,
        source_placements=specs,
        target_mesh=mesh_b,
        target_placements=specs,
        group=dist.group.WORLD,
        device=device,
    )

    # Direction 2: B -> A (reversed to test deadlock)
    m2m_map_b_to_a, source_slicers_b, target_slicers_a = get_m2m_map(
        source_mesh=mesh_b,
        source_placements=specs,
        target_mesh=mesh_a,
        target_placements=specs,
        group=dist.group.WORLD,
        device=device,
    )

    logger.info("Successfully called get_m2m_map twice without deadlock")

    # Generate chunks for the direction this rank participates in
    if is_in_mesh_a:
        # Mesh A sends to Mesh B
        chunks = map_to_chunk_ops(
            m2m_map=m2m_map_a_to_b,
            rank=rank,
            source_num_slicers=source_slicers_a,
            target_num_slicers=target_slicers_b,
            source_tensor=local_tensor,
            target_tensor=None,
        )
    else:
        # Mesh B receives from Mesh A
        chunks = map_to_chunk_ops(
            m2m_map=m2m_map_a_to_b,
            rank=rank,
            source_num_slicers=source_slicers_a,
            target_num_slicers=target_slicers_b,
            source_tensor=None,
            target_tensor=local_tensor,
        )

    logger.info(f"[rank={rank}] Generated {len(chunks)} chunks")

    logger.info(f"[rank={rank}] About to start chunk_comm")

    for chunk in chunks:
        logger.info(f"[rank={rank}] Chunk: {chunk}")

    # Execute communication
    chunk_comm(chunks=chunks)

    logger.info(f"[rank={rank}] Finished chunk_comm")

    # Verify results for receiver side
    if not is_in_mesh_a:
        # The communication wrote to local_tensor, which is part of dist_tensor_b
        # Verify the full tensor matches the source
        full_result = dist_tensor_b.full_tensor()
        assert torch.allclose(full_result, tensor_a_origin), f"M2M communication result mismatch on rank {rank}"

    dist.destroy_process_group()


@pytest.mark.parametrize(
    "mesh_shape",
    [
        (4,),
    ],
)
def test_communication_cpu(mesh_shape: tuple):
    """Test symmetric mesh communication without deadlock."""
    mesh_size = math.prod(mesh_shape)
    world_size = mesh_size * 2  # Two meshes of equal size
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
            args=(world_size, mesh_shape, device),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        pytest.fail(f"Distributed test failed: {e}")
