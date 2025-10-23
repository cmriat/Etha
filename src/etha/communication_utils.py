"""Communication utilities for Etha."""

import itertools

import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, DTensor, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Placement


def get_shard_shape(device_mesh: tuple[int, ...], placements: tuple[Placement, ...], tensor_ndim: int) -> list[int]:
    """Calculate shard shape from device mesh and placements."""
    shard_shape = [1] * tensor_ndim
    for i, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_shape[placement.dim] *= device_mesh[i]
    return shard_shape


def get_shard_tensor_shape(
    origin_full_shape: torch.Size, target_device_mesh: DeviceMesh, placements: tuple[Placement, ...]
) -> torch.Size:
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


def get_slicer_tuples(tensor_shape: torch.Tensor, source_num_slicers: list[int]) -> list[tuple[slice, ...]]:
    slicers_per_dim = []
    for dim, num_slices in enumerate(source_num_slicers):
        dim_size = tensor_shape[dim]
        slice_size = dim_size // num_slices
        slicers_per_dim.append([slice(i * slice_size, (i + 1) * slice_size) for i in range(num_slices)])

    return list(itertools.product(*slicers_per_dim))


def gather_broadcast_communicate(
    target_mesh: DeviceMesh,
    target_specs: tuple[Placement, ...],
    local_tensor: DTensor,
    origin_tensor: torch.Tensor,
    source_world_size: int,
    device: str,
):
    """Performs data redistribution using the Gather-Broadcast method."""
    rank = dist.get_rank()
    gathered_tensor = None
    # 1. Gather the full tensor. After this, every rank in source_mesh has a full copy.
    if rank < source_world_size:
        gathered_tensor = local_tensor.full_tensor()

    # 2. Broadcast the full tensor from a single source (rank 0) to all other ranks.
    # Ranks outside the source_mesh need a placeholder tensor to receive the data.
    if rank >= source_world_size:
        gathered_tensor = torch.empty(origin_tensor.shape, dtype=origin_tensor.dtype, device=device)

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
    forward_map: dict[int, dict[tuple, list[int]]],
    reverse_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    local_tensor: DTensor,
    source_num_slicers: list[int],
    target_num_slicers: list[int],
    target_tensor_shape: torch.Size,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    rank = dist.get_rank()
    send_reqs = []
    recv_reqs = []
    received_data = {}
    # Send data to target ranks based on forward_map
    if rank in forward_map:
        local_tensor = local_tensor.to_local()
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
                    recv_buffer = torch.empty(chunk_shape, dtype=dtype, device=device)
                    req = dist.irecv(tensor=recv_buffer, src=source_rank)
                    recv_reqs.append(req)
                    recv_buffers[(target_idx, source_rank, source_idx)] = recv_buffer

    # Wait for all sends to complete
    for req in send_reqs:
        req.wait()

    # Wait for all receives and assemble final tensor
    final_tensor = None
    if rank in reverse_map:
        final_tensor = torch.empty(target_tensor_shape, dtype=dtype, device=device)

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
