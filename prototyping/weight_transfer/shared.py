"""Shared constants for pair registration demo."""

import os

# Pair configuration
PAIR_NAME = "weights"

# TCPStore configuration
TCPSTORE_HOST = "127.0.0.1"
TCPSTORE_PORT = 29500

# World size (total number of agent ranks)
AGENT_WORLD_SIZE = 8

LMDB_ROOT = f"{os.environ['PIXI_PROJECT_ROOT']}/prototyping/weight_transfer/dbs"


def get_agent_command_queue_path(rank: int) -> str:
    """Get CommandQueue LMDB path for a specific agent rank.

    Args:
        rank: Agent rank (0-7)

    Returns:
        Path to CommandQueue LMDB
    """
    return f"{LMDB_ROOT}/{rank}_command.lmdb"


def get_agent_state_path(rank: int) -> str:
    """Get State LMDB path for a specific agent rank.

    Args:
        rank: Agent rank (0-7)

    Returns:
        Path to State LMDB
    """
    return f"{LMDB_ROOT}/{rank}_state.lmdb"


def get_queue_state_paths(rank: int) -> tuple[str, str]:
    """Get CommandQueue and State LMDB paths for a specific agent rank.

    Args:
        rank: Agent rank (0-7)

    Returns:
        Tuple of CommandQueue and State LMDB paths
    """
    return (get_agent_command_queue_path(rank), get_agent_state_path(rank))
