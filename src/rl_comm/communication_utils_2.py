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
) -> Tuple[Dict[int, List[int]], Dict[int, List[List[int]]], List[int], List[int]]:

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
    local_shard.copy_(torch.full_like(local_shard, rank))
    full_tensor_restored = dtensor_source.full_tensor()
    if rank < source_world_size:
        print(f"Rank {rank} full_tensor_restored: {full_tensor_restored}")
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
    # if rank >= source_world_size:
    #     # Target ranks receive the full tensor
    #     full_tensor_restored = torch.empty(middle_tensor_shape, device=device)

    # dist.broadcast(full_tensor_restored, src=0)

    forward_map = defaultdict(list)
    reverse_map = defaultdict(list)

    if rank >= source_world_size:
        dtensor_target = distribute_tensor(
            full_tensor_restored, target_mesh, target_placements, src_data_rank=None
        )
        print(f"Rank {rank-source_world_size} dtensor_target: {dtensor_target}")
        local_target_shard = dtensor_target.to_local()
        flat_tensor = local_target_shard.flatten()
        for i in range(flat_tensor.numel()):
            source_device = int(flat_tensor[i].item())
            forward_map[source_device].append(rank)
            reverse_map[rank].append(source_device)
    dist.barrier()

    all_forward_maps = [None] * (source_world_size + target_world_size)
    all_reverse_maps = [None] * (source_world_size + target_world_size)

    dist.all_gather_object(all_forward_maps, forward_map, group=None)
    dist.all_gather_object(all_reverse_maps, reverse_map, group=None)

    merged_forward_map = defaultdict(list)
    merged_reverse_map = defaultdict(list)

    for rank_reverse_map in all_reverse_maps:
        if rank_reverse_map is not None:
            for target_rank, source_devices in rank_reverse_map.items():
                merged_reverse_map[target_rank].extend(source_devices)

    for rank_forward_map in all_forward_maps:
        if rank_forward_map is not None:
            for source_device, target_ranks in rank_forward_map.items():
                merged_forward_map[source_device].extend(target_ranks)

    return (
        dict(merged_forward_map),
        dict(merged_reverse_map),
        source_num_slicers,
        target_num_slicers,
    )

def gather_broadcast_communicate(
    rank: int,
    source_mesh: DeviceMesh,
    source_specs: Tuple[Placement, ...],
    target_mesh: DeviceMesh,
    target_specs: Tuple[Placement, ...],
    local_tensor: torch.Tensor,
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
        source_dtensor = DTensor.from_local(local_tensor, source_mesh, source_specs)
        gathered_tensor = source_dtensor.full_tensor()

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
        target_dtensor = distribute_tensor(gathered_tensor, target_mesh, target_specs)
        received_tensor = target_dtensor.to_local()

    return received_tensor

def p2p_communicate(
    rank: int,
    forward_map: Dict[int, List[int]],
    reverse_map: Dict[int, List[List[int]]],
    local_tensor: torch.Tensor,
    source_num_slicers: List[int],
    target_num_slicers: List[int],
    target_tensor_shape: torch.Size,
    device: str = "cpu",
) -> Optional[torch.Tensor]:
    
    received_tensors = {}
    send_reqs = []
    recv_reqs = []

    if rank in forward_map:
        slicer_tuples = get_slicer_tuples(local_tensor.shape, source_num_slicers)
        for i, target_rank in enumerate(forward_map[rank]):
            slice_index = i % len(slicer_tuples)
            chunk_to_send = local_tensor[slicer_tuples[slice_index]]
            if target_rank == rank:
                received_tensors[target_rank] = chunk_to_send
            else:
                req = dist.isend(tensor=chunk_to_send.contiguous(), dst=target_rank)
                send_reqs.append(req)

    if rank in reverse_map:
        expected_chunk_shape = [
            target_tensor_shape[dim] // num_slices
            for dim, num_slices in enumerate(target_num_slicers)
        ]
        for source_rank in reverse_map[rank]:
            if source_rank != rank:
                recv_buffer = torch.empty(expected_chunk_shape, device=device)
                req = dist.irecv(tensor=recv_buffer, src=source_rank)
                recv_reqs.append(req)
                received_tensors[source_rank] = recv_buffer

    # Wait for all operations to complete
    for req in send_reqs:
        req.wait()
    for req in recv_reqs:
        req.wait()

    if rank in reverse_map:
        final_tensor = torch.empty(
            target_tensor_shape,
            dtype=received_tensors[reverse_map[rank][0]].dtype,
            device=received_tensors[reverse_map[rank][0]].device,
        )

        slicer_tuples = get_slicer_tuples(
            target_tensor_shape, target_num_slicers
        )
        for i, source_device in enumerate(reverse_map[rank]):
            slice_tuple = slicer_tuples[i]
            final_tensor[slice_tuple] = received_tensors[source_device]

        return final_tensor

    return None