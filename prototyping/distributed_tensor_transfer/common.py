"""Common constants and configuration for distributed tensor transfer example."""

import re

import torch
from torch.distributed.tensor.placement_types import Shard, Placement, Replicate

# Pair configuration
PAIR_NAME = "distributed_weights"

# World size configuration
AGENT_WORLD_SIZE = 8

# Tensor configuration for example
TENSOR_SHAPE = torch.Size((4, 4))

# Device mesh configurations for different distributed strategies
# Strategy 1: Hybrid DP+MP - replicate on mesh dim 0, shard on mesh dim 1
HYBRID_DP_MP_MESH_SHAPE = (2, 2)  # 2x2 mesh for 4 GPUs
HYBRID_DP_MP_PLACEMENTS = ("Replicate()", "Shard(dim=0)")  # Replicate params on dim 0, shard on dim 0

# Strategy 2: Pure Model Parallel - column-wise sharding of model parameters
PURE_MP_MESH_SHAPE = (4,)  # 1x4 mesh for 4 GPUs
PURE_MP_PLACEMENTS = ("Shard(dim=1)",)  # Shard model parameters on column dimension (512)

# TCPStore configuration
TCPSTORE_HOST = "localhost"
TCPSTORE_PORT = 39505

# Base path for LMDB storage
LMDB_ROOT = "/tmp/dbs"


def get_queue_state_paths(rank: int) -> tuple[str, str]:
    """Get CommandQueue and State LMDB paths for a specific agent rank.

    Args:
        rank: Agent rank (0-7)

    Returns:
        Tuple of CommandQueue and State LMDB paths
    """
    return (f"{LMDB_ROOT}/{rank}_command.lmdb", f"{LMDB_ROOT}/{rank}_state.lmdb")


def get_mesh_config(strategy: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Get device mesh configuration for specified strategy."""
    if strategy == "hybrid_dp_mp":
        return HYBRID_DP_MP_MESH_SHAPE, HYBRID_DP_MP_PLACEMENTS
    elif strategy == "pure_mp":
        return PURE_MP_MESH_SHAPE, PURE_MP_PLACEMENTS
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Available: hybrid_dp_mp, pure_mp")


def read_placement(placement_strs: tuple[str, ...]) -> tuple[Placement, ...]:
    """Read placement strings and return placement objects."""
    placements = []
    for placement_str in placement_strs:
        if "Replicate" in placement_str:
            placements.append(Replicate())
        elif "Shard(dim=" in placement_str:
            dim_match = re.search(r"dim=(\d+)", placement_str)
            if dim_match:
                dim = int(dim_match.group(1))
                placements.append(Shard(dim))
            else:
                raise ValueError(f"Invalid Shard placement: {placement_str}")
        else:
            raise ValueError(f"Unknown placement: {placement_str}")
    return tuple(placements)
