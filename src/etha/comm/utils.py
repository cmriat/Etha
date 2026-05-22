"""Helper functions for communication operations.

Pure utility functions with no side effects.
"""

import itertools

import torch

# Re-export for backward compatibility
from etha.pg_utils import get_or_create_process_group

__all__ = [
    "get_or_create_process_group",
    "get_slicer_tuples",
    "get_slice_from_multi_index",
    "enumerate_partial_subgroup_ranks",
]


def enumerate_partial_subgroup_ranks(
    mesh_tensor: torch.Tensor,
    mesh_dim_idx: int,
) -> list[list[int]]:
    """Sub-group rank lists, one per slice through ``mesh_dim_idx``.

    Each sub-group is the set of ranks that share all coordinates except along
    ``mesh_dim_idx`` — i.e., the ranks that contribute to one all-reduce when
    collapsing a ``Partial`` dim to ``Replicate``.
    """
    ndim = mesh_tensor.dim()
    if not 0 <= mesh_dim_idx < ndim:
        raise ValueError(f"mesh_dim_idx must be in [0, {ndim}), got {mesh_dim_idx}")
    other_dims = [d for d in range(ndim) if d != mesh_dim_idx]
    other_sizes = [mesh_tensor.shape[d] for d in other_dims]
    sub_groups: list[list[int]] = []
    for other_coords in itertools.product(*[range(s) for s in other_sizes]):
        slicer: list = [slice(None)] * ndim
        for di, ci in zip(other_dims, other_coords, strict=False):
            slicer[di] = ci
        sub_groups.append(mesh_tensor[tuple(slicer)].flatten().tolist())
    return sub_groups


def get_slicer_tuples(tensor_shape: torch.Size, source_num_slicers: list[int]) -> list[tuple[slice, ...]]:
    """Pre-compute all slice tuples for a tensor partitioned by num_slicers.

    Args:
        tensor_shape: Shape of the tensor to slice
        source_num_slicers: Number of slices per dimension

    Returns:
        List of slice tuples, one for each chunk
    """
    slicers_per_dim = []
    for dim, num_slices in enumerate(source_num_slicers):
        dim_size = tensor_shape[dim]
        slice_size = dim_size // num_slices
        slicers_per_dim.append([slice(i * slice_size, (i + 1) * slice_size) for i in range(num_slices)])

    return list(itertools.product(*slicers_per_dim))


def get_slice_from_multi_index(
    source_idx: tuple, source_num_slicers: list[int], slicer_tuples: list[tuple[slice, ...]]
) -> tuple[slice, ...]:
    """Convert multi-dimensional index to linear index and return corresponding slice tuple.

    Args:
        source_idx: Multi-dimensional index (e.g., (0, 1))
        source_num_slicers: Number of slices per dimension
        slicer_tuples: Pre-computed slice tuples from get_slicer_tuples()

    Returns:
        Slice tuple for the given index
    """
    linear_idx = 0
    multiplier = 1
    for i in reversed(range(len(source_idx))):
        linear_idx += source_idx[i] * multiplier
        multiplier *= source_num_slicers[i]
    return slicer_tuples[linear_idx]
