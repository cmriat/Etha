"""Process group utilities with caching."""

import torch.distributed as dist

# Global cache for subgroup handles (avoid repeated collective new_group)
_PROCESS_GROUP_CACHE: dict[tuple[int, ...], dist.ProcessGroup] = {}


def get_or_create_process_group(ranks: list[int]) -> dist.ProcessGroup:
    """Get or create a process group for the given ranks.

    Uses caching to avoid repeated dist.new_group() calls.
    Process groups must be created in same order on all ranks (PyTorch requirement).
    The cache key uses sorted ranks to ensure consistency.
    """
    key = tuple(sorted(ranks))
    if key not in _PROCESS_GROUP_CACHE:
        _PROCESS_GROUP_CACHE[key] = dist.new_group(ranks=list(key))
    return _PROCESS_GROUP_CACHE[key]
