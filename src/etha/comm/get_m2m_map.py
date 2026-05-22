"""P2P map for Etha."""

import math
import logging
import itertools
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, Replicate, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Partial, Placement

from .utils import enumerate_partial_subgroup_ranks

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


def _expand_partial_shadows(
    m2m_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    source_mesh: DeviceMesh,
    source_placements: tuple[Placement, ...],
) -> dict[int, dict[tuple, list[tuple[int, tuple]]]]:
    """Insert SHADOW entries so every Partial sub-group member participates in reduce.

    Trace selects one primary per cell; the remaining Partial peers need SHADOW
    entries (empty ``dst_list``) so they reach the chunk-level all-reduce.
    Propagation is transitive: a rank's chunk at cell C drives *all* of its
    sub-groups' reduces, so the whole connected component (sub-groups linked
    via shared members) must be present at C.
    """
    sub_groups: list[list[int]] = []
    mesh_tensor = source_mesh.mesh
    for mesh_dim_idx, p in enumerate(source_placements):
        if isinstance(p, Partial):
            sub_groups.extend(enumerate_partial_subgroup_ranks(mesh_tensor, mesh_dim_idx))
    expanded: dict[int, dict[tuple, list[tuple[int, tuple]]]] = {k: dict(v) for k, v in m2m_map.items()}
    if not sub_groups:
        return expanded

    parent: dict[int, int] = {r: r for sg in sub_groups for r in sg}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for sg in sub_groups:
        anchor = sg[0]
        for r in sg[1:]:
            ra, rr = find(anchor), find(r)
            if ra != rr:
                parent[ra] = rr

    component_members: dict[int, set[int]] = defaultdict(set)
    for r in parent:
        component_members[find(r)].add(r)

    for src_rank, cells in list(m2m_map.items()):
        if src_rank not in parent:
            continue  # rank has no Partial sub-group; nothing to expand
        comp = component_members[find(src_rank)]
        for cell in cells:
            for r in comp:
                if r != src_rank:
                    expanded.setdefault(r, {}).setdefault(cell, [])

    # Sort cells so all component members iterate in lock-step.
    return {r: {cell: expanded[r][cell] for cell in sorted(expanded[r].keys())} for r in expanded}


def get_m2m_map(
    source_mesh: DeviceMesh,
    source_placements: tuple[Placement, ...],
    target_mesh: DeviceMesh,
    target_placements: tuple[Placement, ...],
    group: dist.ProcessGroup,
    device: str = "cpu",
) -> tuple[
    dict[int, dict[tuple, list[tuple[int, tuple]]]],
    list[int],
    list[int],
    list[tuple[int, str]],
]:
    """Get P2P communication map for tensor redistribution.

    Source Partial is supported by substituting Partial→Replicate for the trace,
    then inserting SHADOW entries for the dropped peers via
    ``_expand_partial_shadows``. Target Partial is rejected — the decomposition
    of a logical tensor into Partial contributions is not uniquely defined
    across an independent process-group boundary.

    Returns:
        ``(m2m_map, source_num_slicers, target_num_slicers, source_partial_reductions)``.
        The last is a list of ``(mesh_dim_idx, reduce_op_str)`` per Partial dim,
        empty when source has no Partial.
    """
    if any(isinstance(p, Partial) for p in target_placements):
        raise NotImplementedError(
            f"Partial placement in target_placements={target_placements} is not supported. "
            "Cross-process-group decomposition of a logical tensor into a Partial "
            "contribution is not uniquely defined; only source-side Partial is supported."
        )

    source_partial_reductions: list[tuple[int, str]] = [
        (i, p.reduce_op) for i, p in enumerate(source_placements) if isinstance(p, Partial)
    ]
    effective_source_placements: tuple[Placement, ...] = tuple(
        Replicate() if isinstance(p, Partial) else p for p in source_placements
    )

    rank = dist.get_rank()
    target_mesh_ranks = target_mesh.mesh.flatten().tolist()
    source_mesh_ranks = source_mesh.mesh.flatten().tolist()

    source_tensor_ndim = _get_tensor_ndim(effective_source_placements)
    target_tensor_ndim = _get_tensor_ndim(target_placements)
    tensor_ndim = max(source_tensor_ndim, target_tensor_ndim)

    source_shard_shape = get_shard_shape(source_mesh.mesh.shape, effective_source_placements, tensor_ndim)
    target_shard_shape = get_shard_shape(target_mesh.mesh.shape, target_placements, tensor_ndim)

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
        dtensor_source = distribute_tensor(middle_tensor, source_mesh, effective_source_placements)
        local_shard = dtensor_source.to_local()
        logger.debug(f"[P2P Map rank={rank}] Local Source Shard: {local_shard}")
        encoded_tensor = torch.zeros_like(local_shard)

        # Use middle_tensor_shape to ensure consistent base across all ranks
        base = max(middle_tensor_shape) + 1

        for idx in itertools.product(*[range(dim) for dim in local_shard.shape]):
            encoded_value = rank
            for coord in idx:
                encoded_value = encoded_value * base + coord
            encoded_tensor[idx] = encoded_value

        local_shard.copy_(encoded_tensor)
        full_tensor_restored = dtensor_source.full_tensor()

        source_idx = source_mesh_ranks.index(rank)
        for target_idx in range(source_idx, len(target_mesh_ranks), len(source_mesh_ranks)):
            target_rank = target_mesh_ranks[target_idx]
            logger.debug(f"[P2P Map rank={rank}] Sending to target rank: {target_rank}")
            reqs.append(dist.isend(full_tensor_restored, dst=target_rank))

    elif rank in target_mesh_ranks:
        full_tensor_restored = torch.empty(middle_tensor_shape, device=device)
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
        base = max(middle_tensor_shape) + 1

        for target_idx_tuple in itertools.product(*[range(dim) for dim in local_target_shard.shape]):
            encoded_value = int(local_target_shard[target_idx_tuple].item())

            source_indices = []
            temp = encoded_value
            for _ in range(len(target_idx_tuple)):
                coord = temp % base
                source_indices.append(coord)
                temp = temp // base

            source_rank = temp

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

    final_m2m_map = {}
    for source_rank, source_idx_map in merged_m2m_map.items():
        final_m2m_map[source_rank] = dict(source_idx_map)

    final_m2m_map = _expand_partial_shadows(final_m2m_map, source_mesh, source_placements)

    return final_m2m_map, source_num_slicers, target_num_slicers, source_partial_reductions
