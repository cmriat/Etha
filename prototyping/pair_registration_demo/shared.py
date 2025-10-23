"""Shared constants for pair registration demo."""

# Pair configuration
PAIR_NAME = "obs"

# TCPStore configuration
TCPSTORE_HOST = "127.0.0.1"
TCPSTORE_PORT = 29500

# World size (total number of agent ranks)
AGENT_WORLD_SIZE = 8


def get_agent_command_queue_path(rank: int) -> str:
    """Get CommandQueue LMDB path for a specific agent rank.

    Args:
        rank: Agent rank (0-7)

    Returns:
        Path to CommandQueue LMDB
    """
    return f"/tmp/agent_rank{rank}_command.lmdb"


def get_agent_state_path(rank: int) -> str:
    """Get State LMDB path for a specific agent rank.

    Args:
        rank: Agent rank (0-7)

    Returns:
        Path to State LMDB
    """
    return f"/tmp/agent_rank{rank}_state.lmdb"
