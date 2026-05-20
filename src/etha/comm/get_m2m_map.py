"""P2P map for Etha."""

import math
import logging
import itertools
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Partial, Placement

logger = logging.getLogger(__name__)


def _get_tensor_ndim(placements: tuple[Placement, ...]) -> int:
    """Calculate tensor ndim from placements.

    Returns the number of dimensions based on the maximum Shard dimension + 1.
    """
    return (
        max(
            (placement.dim for placement in placements if isinstance(placement, Shard)),
            default=0,
        )
        + 1
    )


def get_shard_shape(device_mesh: tuple[int, ...], placements: tuple[Placement, ...], tensor_ndim: int) -> list[int]:
    """Calculate shard shape from device mesh and placements."""
    shard_shape = [1] * tensor_ndim
    for i, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_shape[placement.dim] *= device_mesh[i]
    return shard_shape


def get_m2m_map(
    source_mesh: DeviceMesh,
    source_placements: tuple[Placement, ...],
    target_mesh: DeviceMesh,
    target_placements: tuple[Placement, ...],
    group: dist.ProcessGroup,
    device: str = "cpu",
) -> tuple[dict[int, dict[tuple, list[tuple[int, tuple]]]], list[int], list[int]]:
    """Get P2P communication map for tensor redistribution.

    Only ``Shard`` and ``Replicate`` placements are supported. ``Partial`` is
    rejected because the map is built by encoding ``rank`` + ``coord`` into a
    middle tensor and shipping it via ``DTensor.full_tensor()``; for ``Partial``
    that triggers an all-reduce which sums the encoded values across ranks and
    corrupts the map.

    Raises:
        NotImplementedError: If any of ``source_placements`` or
            ``target_placements`` contains a ``Partial``.
    """
    for placements, side in ((source_placements, "source"), (target_placements, "target")):
        if any(isinstance(p, Partial) for p in placements):
            raise NotImplementedError(
                f"Partial placement is not supported (found in {side}_placements={placements}). "
                "Etha currently supports only Shard and Replicate; redistribute Partial to "
                "Replicate or Shard on the source mesh before handing the DTensor to Etha."
            )
    rank = dist.get_rank()
    target_mesh_ranks = target_mesh.mesh.flatten().tolist()
    source_mesh_ranks = source_mesh.mesh.flatten().tolist()

    source_tensor_ndim = _get_tensor_ndim(source_placements)
    target_tensor_ndim = _get_tensor_ndim(target_placements)
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
        dtensor_source = distribute_tensor(middle_tensor, source_mesh, source_placements)
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
        if req is not None:
            req.wait()

    m2m_map: dict[int, dict[tuple, list[tuple[int, tuple]]]] = defaultdict(lambda: defaultdict(list))

    if rank in target_mesh_ranks:
        dtensor_target = distribute_tensor(full_tensor_restored, target_mesh, target_placements, src_data_rank=None)
        local_target_shard = dtensor_target.to_local()
        logger.debug(f"[M2M Map rank={rank}] Local Target Shard: {local_target_shard}")
        # Calculate the same base used for encoding
        # Use middle_tensor_shape to ensure consistent base across all ranks
        base = max(middle_tensor_shape) + 1

        # Iterate through all indices in the local shard
        for target_idx_tuple in itertools.product(*[range(dim) for dim in local_target_shard.shape]):
            encoded_value = int(local_target_shard[target_idx_tuple].item())

            # Decode the rank and source indices from the encoded value
            # Extract coordinates in reverse order
            source_indices = []
            temp = encoded_value
            for _ in range(len(target_idx_tuple)):
                coord = temp % base
                source_indices.append(coord)
                temp = temp // base

            # The remaining value is the source rank
            source_rank = temp

            # Reverse the indices list since we extracted them in reverse order
            source_indices.reverse()
            source_idx = tuple(source_indices)

            target_rank = rank
            m2m_map[source_rank][source_idx].append((target_rank, target_idx_tuple))

    m2m_map_regular = {}
    for k, v in m2m_map.items():
        m2m_map_regular[k] = dict(v)

    all_m2m_maps = [None] * (len(source_mesh_ranks) + len(target_mesh_ranks))
    dist.all_gather_object(all_m2m_maps, m2m_map_regular, group=group)

    merged_m2m_map: dict[int, dict[tuple, list[tuple[int, tuple]]]] = defaultdict(lambda: defaultdict(list))
    for rank_m2m_map in all_m2m_maps:
        if rank_m2m_map is not None:
            for source_rank, source_idx_map in rank_m2m_map.items():
                for source_idx, target_info_list in source_idx_map.items():
                    merged_m2m_map[source_rank][source_idx].extend(target_info_list)

    # Convert nested defaultdict to regular dict
    final_m2m_map = {}
    for source_rank, source_idx_map in merged_m2m_map.items():
        final_m2m_map[source_rank] = dict(source_idx_map)

    return final_m2m_map, source_num_slicers, target_num_slicers
