"""Bootstrap utilities."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from collections.abc import Callable

import torch

from .client import TensorBusClient

logger = logging.getLogger(__name__)

GPU_PER_NODE = 8


@dataclass
class BootstrapInfo:
    """Information about the bootstrap process.

    Attributes:
        agent_rank: The agent rank this worker is connected to
        global_rank: Global rank within the worker process group (from torchrun)
        rank_offset: Offset used for calculation (if applicable)
        device: CUDA device string (e.g., "cuda:0")
        command_queue_path: Path to Agent's CommandQueue LMDB
        state_path: Path to Agent's State LMDB
        method: How agent_rank was determined ("direct" or "offset")
    """

    agent_rank: int
    global_rank: int
    rank_offset: int | None
    device: str
    command_queue_path: str
    state_path: str
    method: str


def bootstrap_client(
    path_naming_fn: Callable[[int], tuple[str, str]] | None = None,
    connection_timeout: float = 30.0,
) -> tuple[TensorBusClient, BootstrapInfo]:
    """Bootstrap TensorBusClient with automatic agent rank resolution.

    This function encapsulates the entire Worker-side bootstrap process:
    1. Determines agent_rank from environment variables (AGENT_RANK or LOCAL_RANK + OFFSET)
    2. Resolves LMDB paths using naming convention
    3. Creates and returns TensorBusClient

    Environment Variables (priority order):
    1. AGENT_RANK: Direct specification (highest priority)
    2. LOCAL_RANK + AGENT_RANK_OFFSET: Offset-based calculation

    Args:
        path_naming_fn: Optional custom function to get (cmd_queue_path, state_path) from rank.
                       If None, uses default convention:
                       - /tmp/agent_rank{N}_command.lmdb
                       - /tmp/agent_rank{N}_state.lmdb
        connection_timeout: Max time to wait for Agent connection (seconds, default 30.0)

    Returns:
        (client, info): Tuple of TensorBusClient and BootstrapInfo

    Raises:
        ValueError: If required environment variables are missing
        ConnectionError: If Agent is not found or not responding
    """
    # Step 1: Determine agent_rank from environment
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if "AGENT_RANK" in os.environ:
        # Priority 1: Direct specification
        rank_offset = None
        agent_rank = int(os.environ["AGENT_RANK"])
        method = "direct"

        logger.info(f"Bootstrap: Using AGENT_RANK={agent_rank} (direct specification)")
    else:
        # Priority 2: Calculate from GLOBAL_RANK + AGENT_RANK_OFFSET
        rank_offset = int(os.environ.get("AGENT_RANK_OFFSET", 0))
        agent_rank = global_rank + rank_offset
        method = "offset"

        logger.info(
            f"Bootstrap: Calculated agent_rank={agent_rank} (GLOBAL_RANK={global_rank} + AGENT_RANK_OFFSET={rank_offset})"
        )

    # Step 2: Resolve LMDB paths
    if path_naming_fn is None:
        # Use default naming convention
        command_queue_path = f"/tmp/agent_rank{agent_rank}_command.lmdb"
        state_path = f"/tmp/agent_rank{agent_rank}_state.lmdb"
    else:
        # Use custom naming function
        command_queue_path, state_path = path_naming_fn(agent_rank)

    logger.info(f"Bootstrap: Resolved paths for Agent {agent_rank}")
    logger.info(f"  CommandQueue: {command_queue_path}")
    logger.info(f"  State: {state_path}")

    # Step 3: Create TensorBusClient
    client = TensorBusClient(
        agent_rank=agent_rank,
        lmdb_command_queue_path=command_queue_path,
        agent_state_lmdb_path=state_path,
        connection_timeout=connection_timeout,
    )

    # Step 4: Create BootstrapInfo
    if rank_offset is not None and rank_offset < GPU_PER_NODE:
        device = f"cuda:{agent_rank}"
        torch.cuda.set_device(device)
    else:
        device = f"cuda:{local_rank}"
    info = BootstrapInfo(
        agent_rank=agent_rank,
        global_rank=global_rank,
        rank_offset=rank_offset,
        device=device,
        command_queue_path=command_queue_path,
        state_path=state_path,
        method=method,
    )

    logger.info(f"Bootstrap: Successfully initialized TensorBusClient (agent_rank={agent_rank}, method={method})")

    return client, info
