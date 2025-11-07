"""Common constants and configuration for distributed tensor transfer example."""

import os

import torch
from torch.distributed.tensor.placement_types import Shard, Replicate

# Pair configuration
PAIR_NAME = "distributed_weights"


# Tensor configuration for example
TENSOR_SHAPE = torch.Size((4, 4))

# Strategy 1: Hybrid DP+MP - replicate on mesh dim 0, shard on mesh dim 1
EXPECTED_WORLD_SIZE = int(os.environ.get("EXPECTED_WORLD_SIZE", 4))
HYBRID_DP_MP_MESH_SHAPE = (2, EXPECTED_WORLD_SIZE // 2)
HYBRID_DP_MP_PLACEMENTS = (Replicate(), Shard(0))  # Replicate params on dim 0, shard on dim 0

# Strategy 2: Pure Model Parallel - row-wise sharding of model parameters
PURE_MP_MESH_SHAPE = (EXPECTED_WORLD_SIZE,)
PURE_MP_PLACEMENTS = (Shard(0),)  # Shard model parameters on row dimension

# Mesh configuration dictionary
MESH_CONFIGS = {
    "hybrid_dp_mp": (HYBRID_DP_MP_MESH_SHAPE, HYBRID_DP_MP_PLACEMENTS),
    "pure_mp": (PURE_MP_MESH_SHAPE, PURE_MP_PLACEMENTS),
}

# TCPStore configuration
TCPSTORE_HOST = os.environ.get("MASTER_ADDR", "localhost")
TCPSTORE_PORT = 40001

# Base path for LMDB storage
LMDB_ROOT = os.environ.get("LMDB_ROOT", "/tmp/dbs")


def get_queue_state_paths(rank: int) -> tuple[str, str]:
    """Get CommandQueue and State LMDB paths for a specific agent rank.

    Args:
        rank: Agent rank

    Returns:
        Tuple of CommandQueue and State LMDB paths
    """
    return (f"{LMDB_ROOT}/{rank}_command.lmdb", f"{LMDB_ROOT}/{rank}_state.lmdb")
