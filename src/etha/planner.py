"""Redistribution planning: placement pair -> routes -> chunks.

``get_m2m_map`` is a pure function of the two sharding declarations — no
process group, no DeviceMesh, no communication. The middle grid (per-dim lcm
of the source and target shard counts) divides evenly by construction, so
every rank's shard is a closed-form box over it; joining the boxes by global
cell id recovers every source cell's destinations. Any rank (or a driver)
computes the identical plan locally.

``get_m2m_map`` 给每条 route 定 ``kind``:单目标 P2P,多目标 BROADCAST。
``m2m_to_chunks`` 把 route 编成每个参与 rank 的 ``Chunk``:源侧出 src_slice(P2P 直发
或 broadcast root),目标侧落 dst_slice(P2P 收或 broadcast 收)。无中继链。

Mesh tensors must hold ranks in the same numbering used at execution time
(the communicator's ranks).
"""

import math
import itertools
from collections import defaultdict

import torch
from torch.distributed.tensor import Shard, Placement, Replicate
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard

_SHARD_TYPES = (Shard, _StridedShard)

from .ir import Cell, Chunk, Route, M2MMap, Endpoint, Transport
from .utils import cell_slice


def _tensor_ndim(placements: tuple[Placement, ...]) -> int:
    return max((p.dim for p in placements if isinstance(p, _SHARD_TYPES)), default=0) + 1


def _shard_counts(mesh_shape: tuple[int, ...], placements: tuple[Placement, ...], tensor_ndim: int) -> list[int]:
    counts = [1] * tensor_ndim
    for i, placement in enumerate(placements):
        if isinstance(placement, _SHARD_TYPES):
            counts[placement.dim] *= mesh_shape[i]
    return counts


def _endpoints(mesh: torch.Tensor, placements: tuple[Placement, ...], middle_shape: tuple[int, ...]):
    """Yield (gid, Endpoint) for every middle cell each rank holds.

    Each rank's box is computed by torch's own sharding geometry
    (``_compute_local_shape_and_global_offset``, the FSDP2/DCP code path) so
    every placement torch can produce — including ``_StridedShard`` — is
    handled by the source of truth, not a reimplementation.
    """
    for coord in itertools.product(*map(range, mesh.shape)):
        span, start = _compute_local_shape_and_global_offset(middle_shape, tuple(mesh.shape), coord, placements)
        rank = int(mesh[coord])
        for cell in itertools.product(*(range(s) for s in span)):
            gid = 0
            for d in range(len(span)):
                gid = gid * middle_shape[d] + start[d] + cell[d]
            yield gid, Endpoint(rank=rank, cell=cell)


def get_m2m_map(
    source_mesh: torch.Tensor,
    source_placements: tuple[Placement, ...],
    target_mesh: torch.Tensor,
    target_placements: tuple[Placement, ...],
) -> M2MMap:
    for placement in (*source_placements, *target_placements):
        if not isinstance(placement, (*_SHARD_TYPES, Replicate)):
            raise NotImplementedError(f"unsupported placement {placement!r}")

    tensor_ndim = max(_tensor_ndim(source_placements), _tensor_ndim(target_placements))
    source_counts = _shard_counts(tuple(source_mesh.shape), source_placements, tensor_ndim)
    target_counts = _shard_counts(tuple(target_mesh.shape), target_placements, tensor_ndim)
    middle_shape = tuple(math.lcm(s, t) for s, t in zip(source_counts, target_counts, strict=True))

    src_holders: dict[int, list[Endpoint]] = defaultdict(list)
    for gid, endpoint in _endpoints(source_mesh, source_placements, middle_shape):
        src_holders[gid].append(endpoint)

    target_index = {rank: i for i, rank in enumerate(target_mesh.flatten().tolist())}
    build: dict[int, dict[Cell, list[Endpoint]]] = defaultdict(lambda: defaultdict(list))
    for gid, dst in _endpoints(target_mesh, target_placements, middle_shape):
        holders = sorted(src_holders[gid])
        holder = holders[target_index[dst.rank] % len(holders)]
        build[holder.rank][holder.cell].append(dst)

    routes = []
    for src_rank in sorted(build):
        cells = build[src_rank]
        for cell in sorted(cells):
            dsts = tuple(cells[cell])
            remote = {d.rank for d in dsts} - {src_rank}  # 落在源 rank 上的 dst 是本地自拷,不计入 wire
            kind = Transport.BROADCAST if len(remote) > 1 else Transport.P2P
            routes.append(Route(src=Endpoint(rank=src_rank, cell=cell), dsts=dsts, kind=kind))
    return M2MMap(
        routes=routes,
        source_num_slicers=[m // s for m, s in zip(middle_shape, source_counts, strict=True)],
        target_num_slicers=[m // t for m, t in zip(middle_shape, target_counts, strict=True)],
    )


def split_fanout(m2m: M2MMap) -> M2MMap:
    """Rewrite one-to-many routes as independent single-destination routes.

    The source then sends every destination its own copy (star fan-out)
    instead of chaining — an A/B switch for benchmarks and an escape hatch.
    """
    return M2MMap(
        routes=[Route(src=route.src, dsts=(dst,)) for route in m2m.routes for dst in route.dsts],
        source_num_slicers=m2m.source_num_slicers,
        target_num_slicers=m2m.target_num_slicers,
    )


def m2m_to_chunks(
    m2m: M2MMap,
    rank: int,
    source_tensor: torch.Tensor | None = None,
    target_tensor: torch.Tensor | None = None,
    target_shape: tuple[int, ...] | None = None,
    transfer_dtype: torch.dtype | None = None,
) -> list[Chunk]:
    if target_tensor is not None:
        target_shape = target_tensor.shape
    chunks: list[Chunk] = []
    for route_idx, route in enumerate(m2m.routes):
        src_rank = route.src.rank
        remote = tuple(sorted({d.rank for d in route.dsts} - {src_rank}))  # wire dst;broadcast 组 = {src}∪remote

        if src_rank == rank:
            src_slice = cell_slice(source_tensor.shape, m2m.source_num_slicers, route.src.cell)
            if remote:  # 源侧 wire:P2P 直发 / broadcast root
                chunks.append(
                    Chunk(
                        route_idx=route_idx,
                        transport=route.kind,
                        src_rank=src_rank,
                        dst_ranks=remote,
                        src_tensor=source_tensor,
                        src_slice=src_slice,
                        transfer_dtype=transfer_dtype,
                    )
                )
            for dst in route.dsts:  # dst 落在源 rank 上:本地自拷,无 wire
                if dst.rank == rank:
                    chunks.append(
                        Chunk(
                            route_idx=route_idx,
                            transport=Transport.LOCAL,
                            src_rank=src_rank,
                            src_tensor=source_tensor,
                            src_slice=src_slice,
                            dst_tensor=target_tensor,
                            dst_slice=cell_slice(target_shape, m2m.target_num_slicers, dst.cell),
                            transfer_dtype=transfer_dtype,
                        )
                    )
        else:
            for dst in route.dsts:  # 目标侧 wire:P2P 收 / broadcast 收
                if dst.rank != rank:
                    continue
                chunks.append(
                    Chunk(
                        route_idx=route_idx,
                        transport=route.kind,
                        src_rank=src_rank,
                        dst_ranks=remote,
                        dst_tensor=target_tensor,
                        dst_slice=cell_slice(target_shape, m2m.target_num_slicers, dst.cell),
                        transfer_dtype=transfer_dtype,
                    )
                )
    return chunks
