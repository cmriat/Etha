import math
import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import Placement

def get_shard_shape(device_mesh: Tuple[int, ...], placements: Tuple[Placement, ...], tensor_ndim: int) -> List[int]:
    """Calculate shard shape from device mesh and placements."""
    shard_shape = [1] * tensor_ndim
    for i, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_shape[placement.dim] *= device_mesh[i]
    return shard_shape

def get_shard_tensor_shape(origin_full_shape: torch.Size, target_device_mesh: DeviceMesh, placements: Tuple[Placement, ...]) -> torch.Size:
    target_shard_shape = list(origin_full_shape)
    for i, placement in enumerate(placements):
        if isinstance(placement, Shard):
            mesh_dim_size = target_device_mesh.mesh.shape[i]
            if target_shard_shape[placement.dim] % mesh_dim_size != 0:
                raise ValueError(
                    f"Dimension {placement.dim} of tensor with shape "
                    f"{target_shard_shape[placement.dim]} is not divisible by "
                    f"mesh dimension {i} with size {mesh_dim_size}"
                )
            target_shard_shape[placement.dim] //= mesh_dim_size
    return torch.Size(target_shard_shape)

def get_slicer_tuples(tensor_shape: torch.Tensor, source_num_slicers: List[int]) -> List[Tuple[slice, ...]]:
    slicers_per_dim = []
    for dim, num_slices in enumerate(source_num_slicers):
        dim_size = tensor_shape[dim]
        slice_size = dim_size // num_slices
        slicers_per_dim.append(
            [slice(i * slice_size, (i + 1) * slice_size) for i in range(num_slices)]
        )

    return list(itertools.product(*slicers_per_dim))
    
def get_p2p_map(
    source_mesh: DeviceMesh,
    source_placements: Tuple[Placement, ...],
    target_mesh: DeviceMesh,
    target_placements: Tuple[Placement, ...],
    rank: int,
    source_world_size: int,
    target_world_size: int,
    device: str = "cpu",
) -> Tuple[Dict[int, Dict[Tuple, List[int]]], Dict[int, Dict[Tuple, List[Tuple[int, Tuple]]]], List[int], List[int]]:

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
    max_coord = max(middle_tensor_shape) if middle_tensor_shape else 1
    base = max_coord + 1  # Base should be larger than any coordinate
    
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
        max_coord = max(middle_tensor_shape) if middle_tensor_shape else 1
        base = max_coord + 1
        
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

def gather_broadcast_communicate(
    rank: int,
    target_mesh: DeviceMesh,
    target_specs: Tuple[Placement, ...],
    local_tensor: DTensor,
    origin_tensor: torch.Tensor,
    source_world_size: int,
    device: str,
):
    """
    Performs data redistribution using the Gather-Broadcast method.
    """
    gathered_tensor = None
    # 1. Gather the full tensor. After this, every rank in source_mesh has a full copy.
    if rank < source_world_size:
        gathered_tensor = local_tensor.full_tensor()

    # 2. Broadcast the full tensor from a single source (rank 0) to all other ranks.
    # Ranks outside the source_mesh need a placeholder tensor to receive the data.
    if rank >= source_world_size:
        gathered_tensor = torch.empty(
            origin_tensor.shape, dtype=origin_tensor.dtype, device=device
        )

    # Rank 0 broadcasts to the default process group (all ranks).
    dist.broadcast(gathered_tensor, src=0)

    # Ensure the broadcast is complete before proceeding.
    dist.barrier()

    # 3. Distribute the now-local full tensor on target ranks.
    received_tensor = None
    if rank >= source_world_size:
        received_tensor = distribute_tensor(gathered_tensor, target_mesh, target_specs)

    return received_tensor

def p2p_communicate(
    rank: int,
    forward_map: Dict[int, Dict[Tuple, List[int]]],
    reverse_map: Dict[int, Dict[Tuple, List[Tuple[int, Tuple]]]],
    local_tensor: torch.Tensor,
    source_num_slicers: List[int],
    target_num_slicers: List[int],
    target_tensor_shape: torch.Size,
    device: str = "cpu",
) -> Optional[torch.Tensor]:
    
    send_reqs = []
    recv_reqs = []
    received_data = {}

    # Send data to target ranks based on forward_map
    if rank in forward_map:
        # Get all possible slice tuples for this local tensor
        slicer_tuples = get_slicer_tuples(local_tensor.shape, source_num_slicers)
        
        for source_idx, target_ranks in forward_map[rank].items():
            # source_idx is like (0,0,1,2), etc. Convert to linear index for slicer_tuples
            # Calculate linear index from multi-dimensional index
            linear_idx = 0
            multiplier = 1
            for i in reversed(range(len(source_idx))):
                linear_idx += source_idx[i] * multiplier
                multiplier *= source_num_slicers[i]
            
            slice_tuple = slicer_tuples[linear_idx]
            data_to_send = local_tensor[slice_tuple]
            
            for target_rank in target_ranks:
                if target_rank == rank:
                    # Store locally if sending to self
                    received_data[(rank, source_idx)] = data_to_send
                else:
                    # Send to other ranks
                    req = dist.isend(tensor=data_to_send.contiguous(), dst=target_rank)
                    send_reqs.append(req)

    # Receive data from source ranks based on reverse_map
    recv_buffers = {}
    if rank in reverse_map:
        # Calculate the size of each chunk being received based on target slicing
        chunk_shape = []
        for dim, num_slices in enumerate(target_num_slicers):
            if num_slices > 1:
                chunk_shape.append(target_tensor_shape[dim] // num_slices)
            else:
                chunk_shape.append(target_tensor_shape[dim])
        chunk_shape = tuple(chunk_shape)
        
        for target_idx, source_info_list in reverse_map[rank].items():
            for source_rank, source_idx in source_info_list:
                if source_rank != rank:
                    # The receive buffer should match the actual chunk size being sent
                    recv_buffer = torch.empty(chunk_shape, dtype=local_tensor.dtype if local_tensor is not None else torch.float32, device=device)
                    req = dist.irecv(tensor=recv_buffer, src=source_rank)
                    recv_reqs.append(req)
                    recv_buffers[(target_idx, source_rank, source_idx)] = recv_buffer

    # Wait for all sends to complete
    for req in send_reqs:
        req.wait()

    # Wait for all receives and assemble final tensor
    final_tensor = None
    if rank in reverse_map:
        final_tensor = torch.empty(target_tensor_shape, dtype=local_tensor.dtype if local_tensor is not None else torch.float32, device=device)
        
        # Wait for all receives to complete
        for req in recv_reqs:
            req.wait()
        
        # Assemble received chunks into final tensor
        for target_idx, source_info_list in reverse_map[rank].items():
            for source_rank, source_idx in source_info_list:
                if source_rank != rank:
                    recv_buffer = recv_buffers[(target_idx, source_rank, source_idx)]
                elif (source_rank, source_idx) in received_data:
                    # Handle local data (self-sends)
                    recv_buffer = received_data[(source_rank, source_idx)]
                else:
                    continue
                
                # Convert target_idx to actual slice coordinates
                # target_idx is like (i, j), convert to slice ranges
                slice_ranges = []
                for dim, coord in enumerate(target_idx):
                    if target_num_slicers[dim] > 1:
                        slice_size = target_tensor_shape[dim] // target_num_slicers[dim]
                        start = coord * slice_size
                        end = start + slice_size
                        slice_ranges.append(slice(start, end))
                    else:
                        slice_ranges.append(slice(None))
                
                slice_tuple = tuple(slice_ranges)
                final_tensor[slice_tuple] = recv_buffer

    return final_tensor