import os
import math
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, DTensor
import json

from rl_comm import get_p2p_map
from rl_comm import p2p_communicate

def get_shard_tensor_shape(origin_full_shape, target_device_mesh, placements):
    target_shard_shape = list(origin_full_shape)
    for i, placement in enumerate(placements):
        if isinstance(placement, Shard):
            mesh_dim_size = target_device_mesh.mesh.shape[i]
            if target_shard_shape[placement.dim] % mesh_dim_size != 0:
                raise ValueError(f"Dimension {placement.dim} of tensor with shape {target_shard_shape[placement.dim]} is not divisible by mesh dimension {i} with size {mesh_dim_size}")
            target_shard_shape[placement.dim] //= mesh_dim_size
    return torch.Size(target_shard_shape)

def run_test():
    """
    Runs a single integration test case passed via environment variables.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Get the test case from environment variables
    case_json = os.environ["TEST_CASE_JSON"]
    test_case = json.loads(case_json)
    
    source_mesh_shape = tuple(test_case["source_mesh"])
    target_mesh_shape = tuple(test_case["target_mesh"])
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    
    source_world_size = math.prod(source_mesh_shape)
    
    if rank == 0:
        print(f"--- Running test: {test_case['name']} ---")
        print(f"Source Mesh: {source_mesh_shape}, Target Mesh: {target_mesh_shape}, Total World Size: {world_size}")

    # Initialize the distributed environment
    dist.init_process_group("gloo")

    # Create separate device meshes for source and target
    source_devices = torch.arange(source_world_size)
    target_devices = torch.arange(source_world_size, world_size)

    source_device_mesh = DeviceMesh(device_type="cpu", mesh=source_devices.reshape(source_mesh_shape))
    target_device_mesh = DeviceMesh(device_type="cpu", mesh=target_devices.reshape(target_mesh_shape))

    # Get P2P maps
    forward_map_local, reverse_map_local = get_p2p_map(source_mesh_shape, placements, target_mesh_shape, placements)

    # Add offset to target ranks to get global ranks
    offset = source_world_size
    forward_map = {
        s_rank: [t_rank + offset for t_rank in t_ranks]
        for s_rank, t_ranks in forward_map_local.items()
    }
    reverse_map = {
        t_rank + offset: s_ranks
        for t_rank, s_ranks in reverse_map_local.items()
    }

    # Create a consistent base tensor for all ranks
    torch.manual_seed(0)
    origin_tensor = torch.randn(64, 64)

    # Determine if the current rank is part of the source or target mesh
    is_in_source = rank < source_world_size
    
    local_tensor_to_send = None
    if is_in_source:
        source_dist_tensor = distribute_tensor(origin_tensor, source_device_mesh, placements)
        local_tensor_to_send = source_dist_tensor.to_local()

    target_local_shape = get_shard_tensor_shape(origin_tensor.shape, target_device_mesh, placements)
    origin_local_shape = get_shard_tensor_shape(origin_tensor.shape, source_device_mesh, placements)
    # All ranks participate in communication
    received_tensor = p2p_communicate(rank, forward_map, reverse_map, local_tensor_to_send, origin_local_shape, target_local_shape)

    # Verification
    if not is_in_source: # Ranks in target mesh should have received a tensor
        assert received_tensor is not None, f"Rank {rank} in target mesh did not receive a tensor."
        
        ground_truth_dist_tensor = distribute_tensor(origin_tensor, target_device_mesh, placements)
        received_dist_tensor = DTensor.from_local(received_tensor, target_device_mesh, placements)
        received_dist_tensor_full = received_dist_tensor.full_tensor()
        assert torch.allclose(received_dist_tensor.to_local(), ground_truth_dist_tensor.to_local()), "Received local tensor doesn't match ground truth"
        assert torch.allclose(received_dist_tensor_full, origin_tensor), "Received full tensor doesn't match original tensor"

    dist.barrier()
    if rank == 0:
        print(f"--- Test PASSED: {test_case['name']} ---")

    dist.destroy_process_group()

if __name__ == "__main__":
    run_test()