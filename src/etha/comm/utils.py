"""Helper functions for communication operations.

Pure utility functions with no side effects (except process group caching).
"""

import itertools

import torch
import torch.distributed as dist

# Global cache for subgroup handles (avoid repeated collective new_group)
_PROCESS_GROUP_CACHE: dict[tuple[int, ...], dist.ProcessGroup] = {}


def get_or_create_process_group(ranks: list[int]) -> dist.ProcessGroup:
    """Get or create a process group for the given ranks.

    Uses caching to avoid repeated dist.new_group() calls.
    Process groups must be created in same order on all ranks (PyTorch requirement).
    """
    key = tuple(sorted(ranks))
    if key not in _PROCESS_GROUP_CACHE:
        _PROCESS_GROUP_CACHE[key] = dist.new_group(ranks=list(key))
    return _PROCESS_GROUP_CACHE[key]


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
