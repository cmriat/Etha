"""Cell-to-slice mapping and sharding geometry helpers."""

import torch
from torch.distributed.tensor import Placement
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset


def cell_slice(shape: torch.Size, num_slicers: list[int], cell: tuple[int, ...]) -> tuple[slice, ...]:
    ns = (num_slicers + [1] * len(shape))[: len(shape)]
    cell = (*cell, *(0,) * (len(shape) - len(cell)))
    return tuple(slice(c * (s // n), (c + 1) * (s // n)) for s, n, c in zip(shape, ns, cell, strict=True))


def local_shape(
    global_shape: tuple[int, ...], mesh: torch.Tensor, placements: tuple[Placement, ...], rank: int
) -> tuple[int, ...]:
    """The shard shape ``rank`` holds under the declared sharding (planner geometry)."""
    coord = tuple((mesh == rank).nonzero()[0].tolist())
    shape, _ = _compute_local_shape_and_global_offset(global_shape, tuple(mesh.shape), coord, placements)
    return shape
