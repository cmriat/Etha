"""P2P Communication Mapping for Distributed Tensor Operations."""

from collections import defaultdict
import itertools
import math
from typing import Dict, List, Tuple
import torch
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard


def get_device_id_from_coords(coords: Tuple[int, ...], shard_shape: List[int], tensor_shape: Tuple[int, ...]) -> int:
    """Convert multi-dimensional coordinates to flat device ID."""
    num_dims = len(coords)
    
    # Calculate grid shape (number of shards per dimension)
    grid_shape = [tensor_shape[i] // shard_shape[i] for i in range(num_dims)]
    
    # Calculate shard coordinates
    shard_coords = [coords[i] // grid_shape[i] for i in range(num_dims)]
    
    # Convert to flat device ID (row-major order)
    device_id = 0
    multiplier = 1
    for i in range(num_dims - 1, -1, -1):
        device_id += shard_coords[i] * multiplier
        multiplier *= shard_shape[i]
    
    return device_id


def get_p2p_map(
    origin_device_mesh: Tuple[int, ...], 
    origin_placements: Tuple[Placement, ...], 
    target_device_mesh: Tuple[int, ...], 
    target_placements: Tuple[Placement, ...]
) -> Tuple[Dict[int, List[int]], Dict[int, List[List[int]]]]:
    """Generate P2P communication mapping for tensor redistribution."""
    
    def get_shard_shape(device_mesh: Tuple[int, ...], placements: Tuple[Placement, ...]) -> List[int]:
        """Calculate shard shape from device mesh and placements."""
        shard_shape = [1, 1]
        for i, placement in enumerate(placements):
            if isinstance(placement, Shard):
                shard_shape[placement.dim] *= device_mesh[i]
        return shard_shape
    
    # Calculate shard shapes
    origin_shard_shape = get_shard_shape(origin_device_mesh, origin_placements)
    target_shard_shape = get_shard_shape(target_device_mesh, target_placements)
    
    # Calculate replica and shard counts
    origin_replica_count = math.prod([
        mesh_dim if isinstance(placement, Replicate) else 1 
        for mesh_dim, placement in zip(origin_device_mesh, origin_placements)
    ])
    target_replica_count = math.prod([
        mesh_dim if isinstance(placement, Replicate) else 1 
        for mesh_dim, placement in zip(target_device_mesh, target_placements)
    ])
    origin_shard_count = math.prod([
        mesh_dim if isinstance(placement, Shard) else 1 
        for mesh_dim, placement in zip(origin_device_mesh, origin_placements)
    ])
    target_shard_count = math.prod([
        mesh_dim if isinstance(placement, Shard) else 1 
        for mesh_dim, placement in zip(target_device_mesh, target_placements)
    ])
    
    # Calculate temporary tensor shape
    temp_tensor_shape = tuple(
        math.lcm(origin_shard_shape[i], target_shard_shape[i]) 
        for i in range(len(origin_shard_shape))
    )
    
    temp_tensor = torch.zeros(temp_tensor_shape, dtype=torch.int32)
    for coords in itertools.product(*[range(dim) for dim in temp_tensor_shape]):
        temp_tensor[coords] = get_device_id_from_coords(coords, origin_shard_shape, temp_tensor_shape)

    # Build communication mappings
    forward_map = defaultdict(list)
    reverse_map = defaultdict(lambda: [[] for _ in range(temp_tensor_shape[0])])
    for replica_id in range(target_replica_count):
        for coords in itertools.product(*[range(dim) for dim in temp_tensor_shape]):
            origin_device_id = temp_tensor[coords]
            target_device_id = get_device_id_from_coords(coords, target_shard_shape, temp_tensor_shape)
            
            # Apply replica offsets
            target_device_id = int(target_device_id + replica_id * target_shard_count)
            origin_device_id = int(origin_device_id + (replica_id % origin_replica_count) * origin_shard_count)
            
            forward_map[origin_device_id].append(target_device_id)
            reverse_map[target_device_id][coords[0]].append(origin_device_id)
    return dict(forward_map), dict(reverse_map)


def print_p2p_summary(forward_map: Dict[int, List[int]], reverse_map: Dict[int, List[List[int]]]) -> None:
    """Print formatted P2P communication summary."""
    print("\n" + "=" * 50)
    print("P2P Communication Mapping Results")
    print("(Source -> Target)")
    print("=" * 50)
    print("Format: Source Device ID -> [Target Device ID List]")
    
    for source, targets in sorted(forward_map.items()):
        print(f"Source Device {source} -> Sends to {len(targets)} target devices: {targets}")
    
    print("\n" + "=" * 50)
    print("(Target -> Source)")
    print("=" * 50)
    print("Format: Target Device ID -> [Source Device ID List]")
    
    for target, sources in sorted(reverse_map.items()):
        num_sources = sum(len(src_list) for src_list in sources)
        print(f"Target Device {target} -> Receives from {num_sources} source devices: {sources}")


if __name__ == "__main__":
    # Example usage [cp, dp_replicate, pp, dp_shard, tp]
    origin_device_mesh = (1, 2, 2, 1, 2)
    target_device_mesh = (1, 4, 1, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))

    forward_map, reverse_map = get_p2p_map(origin_device_mesh, placements, target_device_mesh, placements)
    print_p2p_summary(forward_map, reverse_map)
    assert forward_map == {
        0: [0, 4],
        1: [1, 5],
        2: [0, 4],
        3: [1, 5],
        4: [2, 6],
        5: [3, 7],
        6: [2, 6],
        7: [3, 7],
    }
    assert reverse_map == {
        0: [[0], [2]],
        1: [[1], [3]],
        2: [[4], [6]],
        3: [[5], [7]],
        4: [[0], [2]],
        5: [[1], [3]],
        6: [[4], [6]],
        7: [[5], [7]],
    }
    print("Test passed successfully!")