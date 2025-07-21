import math
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import Placement

from .communication_utils import get_shard_shape


def get_p2p_map(
    source_mesh: DeviceMesh,
    source_placements: Tuple[Placement, ...],
    target_mesh: DeviceMesh,
    target_placements: Tuple[Placement, ...],
    device: str = "cpu",
) -> Tuple[Dict[int, Dict[Tuple, List[int]]], Dict[int, Dict[Tuple, List[Tuple[int, Tuple]]]], List[int], List[int]]:
    rank = dist.get_rank()
    source_world_size = source_mesh.size()
    target_world_size = target_mesh.size()

    source_tensor_ndim = max(
        (placement.dim for placement in source_placements if isinstance(placement, Shard)),
        default=-1,
    ) + 1
    target_tensor_ndim = max(
        (placement.dim for placement in target_placements if isinstance(placement, Shard)),
        default=-1,
    ) + 1
    tensor_ndim = max(source_tensor_ndim, target_tensor_ndim)

    # Calculate shard shapes
    source_shard_shape = get_shard_shape(
        source_mesh.mesh.shape, source_placements, tensor_ndim
    )
    target_shard_shape = get_shard_shape(
        target_mesh.mesh.shape, target_placements, tensor_ndim
    )

    # Calculate temporary tensor shape
    middle_tensor_shape = tuple(
        math.lcm(source_shard_shape[i], target_shard_shape[i])
        for i in range(len(source_shard_shape))
    )
    # Always use CPU for mapping computation regardless of actual communication device
    middle_tensor = torch.zeros(middle_tensor_shape, device=device)
    source_num_slicers = []
    target_num_slicers = []
    for o, m, t in zip(source_shard_shape, middle_tensor_shape, target_shard_shape):
        if o < m:
            source_num_slicers.append(m // o)
        else:
            source_num_slicers.append(1)
        target_num_slicers.append(m // t)

    dtensor_source = distribute_tensor(middle_tensor, source_mesh, source_placements)
    local_shard = dtensor_source.to_local()
    
    # Create tensor with rank and coordinates encoded as single values
    # Use dynamic base calculation to support arbitrary dimensions
    encoded_tensor = torch.zeros_like(local_shard)
    
    # Calculate the maximum coordinate in any dimension to determine encoding base
    # Use middle_tensor_shape to ensure consistent base across all ranks
    base = max(middle_tensor_shape) + 1  # Base should be larger than any coordinate
    
    for idx in itertools.product(*[range(dim) for dim in local_shard.shape]):
        # Encode rank and coordinates into single value using dynamic base
        encoded_value = rank
        for coord in idx:
            encoded_value = encoded_value * base + coord
        encoded_tensor[idx] = encoded_value
    
    local_shard.copy_(encoded_tensor)
    full_tensor_restored = dtensor_source.full_tensor()
    dist.barrier()
    if rank < source_world_size:
        # Source ranks send to target ranks
        send_requests = []
        for target_rank in range(
            source_world_size + rank, source_world_size + target_world_size, source_world_size
        ):
            req = dist.isend(full_tensor_restored, dst=target_rank)
            send_requests.append(req)

        # Wait for all sends to complete
        for req in send_requests:
            req.wait()
    else:
        # Target ranks receive from source ranks
        full_tensor_restored = torch.empty(middle_tensor_shape, device=device)
        source_rank = (rank - source_world_size) % source_world_size
        req = dist.irecv(full_tensor_restored, src=source_rank)
        req.wait()

    def make_nested_defaultdict():
        return defaultdict(list)
    
    forward_map = defaultdict(make_nested_defaultdict)
    reverse_map = defaultdict(make_nested_defaultdict)

    if rank >= source_world_size:
        dtensor_target = distribute_tensor(
            full_tensor_restored, target_mesh, target_placements, src_data_rank=None
        )
        local_target_shard = dtensor_target.to_local()
        
        # Calculate the same base used for encoding
        # Use middle_tensor_shape to ensure consistent base across all ranks
        base = max(middle_tensor_shape) + 1
        
        # Iterate through all indices in the local shard
        for target_idx in itertools.product(*[range(dim) for dim in local_target_shard.shape]):
            encoded_value = int(local_target_shard[target_idx].item())
            
            # Decode the rank and source indices from the encoded value
            # Extract coordinates in reverse order
            source_indices = []
            temp = encoded_value
            for _ in range(len(target_idx)):
                coord = temp % base
                source_indices.append(coord)
                temp = temp // base
            
            # The remaining value is the source rank
            source_rank = temp
            
            # Reverse the indices list since we extracted them in reverse order
            source_indices.reverse()
            source_idx = tuple(source_indices)
            
            # Build forward_map: source_rank -> {source_idx: [target_ranks]}
            target_rank = rank
            forward_map[source_rank][source_idx].append(target_rank)
            
            # Build reverse_map: target_rank -> {target_idx: [(source_rank, source_idx)]}
            reverse_map[target_rank][target_idx].append((source_rank, source_idx))
    dist.barrier()

    # Convert defaultdict to regular dict for serialization
    forward_map_regular = {}
    for k, v in forward_map.items():
        forward_map_regular[k] = dict(v)
    
    reverse_map_regular = {}
    for k, v in reverse_map.items():
        reverse_map_regular[k] = dict(v)

    all_forward_maps = [None] * (source_world_size + target_world_size)
    all_reverse_maps = [None] * (source_world_size + target_world_size)

    dist.all_gather_object(all_forward_maps, forward_map_regular, group=None)
    dist.all_gather_object(all_reverse_maps, reverse_map_regular, group=None)

    merged_forward_map = defaultdict(make_nested_defaultdict)
    merged_reverse_map = defaultdict(make_nested_defaultdict)

    for rank_reverse_map in all_reverse_maps:
        if rank_reverse_map is not None:
            for target_rank, target_idx_map in rank_reverse_map.items():
                for target_idx, source_info_list in target_idx_map.items():
                    merged_reverse_map[target_rank][target_idx].extend(source_info_list)

    for rank_forward_map in all_forward_maps:
        if rank_forward_map is not None:
            for source_rank, source_idx_map in rank_forward_map.items():
                for source_idx, target_ranks in source_idx_map.items():
                    merged_forward_map[source_rank][source_idx].extend(target_ranks)

    # Convert nested defaultdict to regular dict
    final_forward_map = {}
    for source_rank, source_idx_map in merged_forward_map.items():
        final_forward_map[source_rank] = dict(source_idx_map)
    
    final_reverse_map = {}
    for target_rank, target_idx_map in merged_reverse_map.items():
        final_reverse_map[target_rank] = dict(target_idx_map)
    
    return (
        final_forward_map,
        final_reverse_map,
        source_num_slicers,
        target_num_slicers,
    )
