"""Communication utilities for Etha."""

import logging
import itertools
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, DTensor, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Placement

logger = logging.getLogger(__name__)

# Global cache for subgroup handles (avoid repeated collective new_group)
_PROCESS_GROUP_CACHE: dict[tuple[int, ...], dist.ProcessGroup] = {}


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


def get_or_create_process_group(ranks: list[int]) -> dist.ProcessGroup:
    key = tuple(sorted(ranks))
    if key not in _PROCESS_GROUP_CACHE:
        _PROCESS_GROUP_CACHE[key] = dist.new_group(ranks=list(key))
    return _PROCESS_GROUP_CACHE[key]


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


def get_slice_from_multi_index(
    source_idx: tuple, source_num_slicers: list[int], slicer_tuples: list[tuple[slice, ...]]
) -> tuple[slice, ...]:
    """Convert multi-dimensional index to linear index and return corresponding slice tuple."""
    linear_idx = 0
    multiplier = 1
    for i in reversed(range(len(source_idx))):
        linear_idx += source_idx[i] * multiplier
        multiplier *= source_num_slicers[i]
    return slicer_tuples[linear_idx]


def build_broadcast_plan(
    forward_map: dict[int, dict[tuple, list[int]]],
) -> tuple[dict[tuple[int, tuple[int, ...]], list[tuple]], set[tuple[int, tuple]]]:
    """Build broadcast plan for 1-to-many transfers.

    Returns:
        broadcast_plan: Maps (src, tuple(targets)) -> [source_idx...]
        broadcast_keys: Set of (src, source_idx) that use broadcast
    """
    broadcast_plan: dict[tuple[int, tuple[int, ...]], list[tuple]] = defaultdict(list)
    for src in sorted(forward_map.keys()):
        inner = forward_map[src]
        for source_idx in sorted(inner.keys()):
            targets = inner[source_idx]
            other_targets = sorted([r for r in targets if r != src])
            if len(other_targets) > 1:
                broadcast_plan[(src, tuple(other_targets))].append(source_idx)

    broadcast_keys: set[tuple[int, tuple]] = set()
    for (src, _targets), idx_list in broadcast_plan.items():
        for sx in idx_list:
            broadcast_keys.add((src, sx))

    # Create all subgroups in a consistent order across all ranks
    for group_key in sorted(broadcast_plan.keys(), key=lambda k: (k[0], k[1])):
        group_ranks = [group_key[0]] + list(group_key[1])
        get_or_create_process_group(group_ranks)

    return broadcast_plan, broadcast_keys


def assemble_target_tensor(
    rank: int,
    reverse_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    received_data: dict[tuple[int, tuple], torch.Tensor],
    recv_buffers: dict[tuple[tuple, int, tuple], torch.Tensor],
    target_tensor_shape: torch.Size,
    target_num_slicers: list[int],
    final_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Assemble final tensor from received chunks.

    Returns:
        Final assembled tensor if rank is a receiver, None otherwise.
    """
    if rank not in reverse_map:
        return None

    for target_idx, src_list in reverse_map[rank].items():
        for src_rank, sidx in src_list:
            if src_rank == rank:
                if (src_rank, sidx) not in received_data:
                    continue
                buf = received_data[(src_rank, sidx)]
            else:
                buf_key = (target_idx, src_rank, sidx)
                buf = recv_buffers[buf_key]

            slice_ranges = []
            for dim, coord in enumerate(target_idx):
                if target_num_slicers[dim] > 1:
                    slice_size = target_tensor_shape[dim] // target_num_slicers[dim]
                    start = coord * slice_size
                    end = start + slice_size
                    slice_ranges.append(slice(start, end))
                else:
                    slice_ranges.append(slice(None))

            final_tensor[tuple(slice_ranges)].copy_(buf)

            if src_rank != rank:
                del recv_buffers[buf_key]


def p2p_communicate(
    source_local_tensor: torch.Tensor,
    target_local_tensor: torch.Tensor,
    forward_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    reverse_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    source_num_slicers: list[int],
    target_num_slicers: list[int],
) -> torch.Tensor | None:
    rank = dist.get_rank()

    # Build broadcast plan for 1-to-many transfers
    broadcast_plan, broadcast_keys = build_broadcast_plan(forward_map)

    p2p_send_ops = []
    p2p_recv_ops = []
    bcast_send_works = []
    bcast_recv_works = []

    received_data: dict[tuple[int, tuple], torch.Tensor] = {}
    recv_buffers: dict[tuple[tuple, int, tuple], torch.Tensor] = {}

    slicer_tuples = None
    chunk_shape = None
    target_tensor_shape = target_local_tensor.shape
    if rank in forward_map:
        slicer_tuples = get_slicer_tuples(source_local_tensor.shape, source_num_slicers)
    else:
        chunk_shape = tuple(
            target_tensor_shape[dim] // num_slices if num_slices > 1 else target_tensor_shape[dim]
            for dim, num_slices in enumerate(target_num_slicers)
        )

    # Self-sends: store locally
    if rank in forward_map and source_local_tensor is not None:
        for source_idx, target_ranks in forward_map[rank].items():
            if rank in target_ranks:
                slice_tuple = get_slice_from_multi_index(source_idx, source_num_slicers, slicer_tuples)
                received_data[(rank, source_idx)] = source_local_tensor[slice_tuple]

    # Async broadcasts (send/recv) in deterministic order per group
    for group_key in sorted(broadcast_plan.keys(), key=lambda k: (k[0], k[1])):
        src_rank, other_targets = group_key[0], list(group_key[1])
        group = get_or_create_process_group([src_rank] + other_targets)
        if rank == src_rank:
            for source_idx in broadcast_plan[group_key]:
                slice_tuple = get_slice_from_multi_index(source_idx, source_num_slicers, slicer_tuples)
                data_slice = source_local_tensor[slice_tuple]
                # Only create contiguous copy if necessary for cross-process transfer
                data_to_send = data_slice if data_slice.is_contiguous() else data_slice.contiguous()
                w = dist.broadcast(tensor=data_to_send, src=src_rank, group=group, async_op=True)
                bcast_send_works.append(w)
        elif rank in other_targets:
            if rank in reverse_map:
                for target_idx, src_list in reverse_map[rank].items():
                    for src_rank_check, source_idx in src_list:
                        # Only process broadcasts for current group
                        if src_rank_check == src_rank and source_idx in broadcast_plan[group_key]:
                            recv_buffer = torch.empty(chunk_shape, dtype=target_local_tensor.dtype, device=target_local_tensor.device)
                            w = dist.broadcast(tensor=recv_buffer, src=src_rank, group=group, async_op=True)
                            bcast_recv_works.append(w)
                            recv_buffers[(target_idx, src_rank_check, source_idx)] = (
                                recv_buffer  # TODO: copy send pipeline
                            )

    # Batched P2P for single-target transfers
    if rank in forward_map and source_local_tensor is not None:
        for source_idx, target_ranks in sorted(forward_map[rank].items()):
            if (rank, source_idx) in broadcast_keys:
                continue
            other_targets = [r for r in target_ranks if r != rank]
            if len(other_targets) == 1:
                slice_tuple = get_slice_from_multi_index(source_idx, source_num_slicers, slicer_tuples)
                data_slice = source_local_tensor[slice_tuple]
                # Only create contiguous copy if necessary for cross-process transfer
                data_to_send = data_slice if data_slice.is_contiguous() else data_slice.contiguous()
                p2p_send_ops.append(dist.P2POp(dist.isend, data_to_send, other_targets[0]))

    if rank in reverse_map:
        for target_idx, src_list in reverse_map[rank].items():
            for src_rank, sidx in src_list:
                if src_rank == rank:
                    continue
                if (src_rank, sidx) in broadcast_keys:
                    continue
                recv_buffer = torch.empty(chunk_shape, dtype=target_local_tensor.dtype, device=target_local_tensor.device)
                p2p_recv_ops.append(dist.P2POp(dist.irecv, recv_buffer, src_rank))
                recv_buffers[(target_idx, src_rank, sidx)] = recv_buffer

    # Launch P2P batches and wait
    send_reqs = dist.batch_isend_irecv(p2p_send_ops) if p2p_send_ops else []
    recv_reqs = dist.batch_isend_irecv(p2p_recv_ops) if p2p_recv_ops else []

    for r in recv_reqs:
        r.wait()
    for r in send_reqs:
        r.wait()
    for w in bcast_recv_works:
        w.wait()
    for w in bcast_send_works:
        w.wait()

    return assemble_target_tensor(
        rank, reverse_map, received_data, recv_buffers, target_tensor_shape, target_num_slicers, target_local_tensor
    )
