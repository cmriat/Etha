"""P2P map for Etha."""

import math
import logging
import itertools
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Placement

from .communication_utils import get_shard_shape

logger = logging.getLogger(__name__)


def get_p2p_map(
    source_mesh: DeviceMesh,
    source_placements: tuple[Placement, ...],
    target_mesh: DeviceMesh,
    target_placements: tuple[Placement, ...],
    device: str = "cpu",
) -> tuple[dict[int, dict[tuple, list[int]]], dict[int, dict[tuple, list[tuple[int, tuple]]]], list[int], list[int]]:
    """Get P2P communication map for tensor redistribution."""
    rank = dist.get_rank()
    target_mesh_ranks = target_mesh.mesh.flatten().tolist()
    source_mesh_ranks = source_mesh.mesh.flatten().tolist()

    source_tensor_ndim = (
        max(
            (placement.dim for placement in source_placements if isinstance(placement, Shard)),
            default=-1,
        )
        + 1
    )
    target_tensor_ndim = (
        max(
            (placement.dim for placement in target_placements if isinstance(placement, Shard)),
            default=-1,
        )
        + 1
    )
    tensor_ndim = max(source_tensor_ndim, target_tensor_ndim)

    # Calculate shard shapes
    source_shard_shape = get_shard_shape(source_mesh.mesh.shape, source_placements, tensor_ndim)
    target_shard_shape = get_shard_shape(target_mesh.mesh.shape, target_placements, tensor_ndim)

    # Calculate temporary tensor shape
    middle_tensor_shape = tuple(
        math.lcm(source_shard_shape[i], target_shard_shape[i]) for i in range(len(source_shard_shape))
    )
    logger.debug(
        f"[P2P Map rank={rank}] Source/Target/Middle Shard Shape: {source_shard_shape} {target_shard_shape} {middle_tensor_shape}"
    )
    source_num_slicers = []
    target_num_slicers = []
    for o, m, t in zip(source_shard_shape, middle_tensor_shape, target_shard_shape, strict=False):
        source_num_slicers.append(m // o)
        target_num_slicers.append(m // t)
    reqs = []
    if rank in source_mesh_ranks:
        middle_tensor = torch.zeros(middle_tensor_shape, device=device)
        dtensor_source = distribute_tensor(middle_tensor, source_mesh, source_placements, src_data_rank=None)
        local_shard = dtensor_source.to_local()
        logger.debug(f"[P2P Map rank={rank}] Local Source Shard: {local_shard}")
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

        # Find index in source mesh
        source_idx = source_mesh_ranks.index(rank)
        # Map to target ranks using original pattern: start from source_idx, step by source mesh size
        for target_idx in range(source_idx, len(target_mesh_ranks), len(source_mesh_ranks)):
            target_rank = target_mesh_ranks[target_idx]
            logger.debug(f"[P2P Map rank={rank}] Sending to target rank: {target_rank}")
            reqs.append(dist.isend(full_tensor_restored, dst=target_rank))

    elif rank in target_mesh_ranks:
        # Target ranks receive from source ranks
        full_tensor_restored = torch.empty(middle_tensor_shape, device=device)
        # Find index in target mesh and map to corresponding source rank
        target_idx = target_mesh_ranks.index(rank)
        source_rank = source_mesh_ranks[target_idx % len(source_mesh_ranks)]

        logger.debug(f"[P2P Map rank={rank}] Receiving from source rank: {source_rank}")
        reqs.append(dist.irecv(full_tensor_restored, src=source_rank))

    for req in reqs:
        req.wait()

    def make_nested_defaultdict():
        return defaultdict(list)

    forward_map = defaultdict(make_nested_defaultdict)
    reverse_map = defaultdict(make_nested_defaultdict)

    if rank in target_mesh_ranks:
        dtensor_target = distribute_tensor(full_tensor_restored, target_mesh, target_placements, src_data_rank=None)
        local_target_shard = dtensor_target.to_local()
        logger.debug(f"[P2P Map rank={rank}] Local Target Shard: {local_target_shard}")
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

            # Build forward_map: source_rank -> {source_idx: [(target_rank, target_idx)]}
            target_rank = rank
            forward_map[source_rank][source_idx].append((target_rank, target_idx))
            # Build reverse_map: target_rank -> {target_idx: [(source_rank, source_idx)]}
            reverse_map[target_rank][target_idx].append((source_rank, source_idx))

    # Convert defaultdict to regular dict for serialization
    forward_map_regular = {}
    for k, v in forward_map.items():
        forward_map_regular[k] = dict(v)

    reverse_map_regular = {}
    for k, v in reverse_map.items():
        reverse_map_regular[k] = dict(v)

    all_forward_maps = [None] * (len(source_mesh_ranks) + len(target_mesh_ranks))
    all_reverse_maps = [None] * (len(source_mesh_ranks) + len(target_mesh_ranks))

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
                for source_idx, target_info_list in source_idx_map.items():
                    merged_forward_map[source_rank][source_idx].extend(target_info_list)

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
