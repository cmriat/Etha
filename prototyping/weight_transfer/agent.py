"""Agent Process - Pair Registration Demo.

Launch with torchrun:
    torchrun --nproc_per_node=8 prototyping/pair_registration_demo/daemon.py

This will start 8 Agent processes that:
1. Form a torch.distributed NCCL group
2. Share a TCPStore for metadata exchange
3. Each agent polls its own CommandQueue for RegisterPair messages
4. Execute pair matching via TCPStore
5. Write PairState to State LMDB when matched
"""

import os
import logging
from pathlib import Path

import torch.distributed as dist
from shared import (
    TCPSTORE_HOST,
    TCPSTORE_PORT,
    AGENT_WORLD_SIZE,
    get_agent_state_path,
    get_agent_command_queue_path,
)

from etha.tensor_bus import TensorBusAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed to DEBUG for detailed logs
    format="[%(asctime)s] [Agent %(process)d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    # Get rank and world_size from environment (torchrun sets these)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", AGENT_WORLD_SIZE))

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Agent Rank {rank} starting...")
    logger.info(f"World Size: {world_size}")
    logger.info(f"TCPStore: {TCPSTORE_HOST}:{TCPSTORE_PORT}")
    logger.info(f"{'=' * 60}\n")

    # Get LMDB paths
    command_queue_path = get_agent_command_queue_path(rank)
    state_path = get_agent_state_path(rank)

    # Clean up old LMDB files (remove stale data and locks)
    for path_str in [command_queue_path, state_path]:
        path = Path(path_str)
        for f in path.parent.glob(f"{path.name}*"):
            f.unlink(missing_ok=True)
            logger.debug(f"[Agent {rank}] Cleaned up: {f}")

    logger.info(f"[Agent {rank}] CommandQueue: {command_queue_path}")
    logger.info(f"[Agent {rank}] State LMDB: {state_path}\n")

    # Initialize Agent
    agent = TensorBusAgent(
        rank=rank,
        world_size=world_size,
        tcpstore_host=TCPSTORE_HOST,
        tcpstore_port=TCPSTORE_PORT,
        lmdb_command_queue_path=command_queue_path,
        lmdb_state_path=state_path,
    )

    logger.info(f"[Agent {rank}] ✅ Initialized successfully")
    logger.info(f"[Agent {rank}] Entering main loop (polling for commands)...\n")

    # Wait for all agents to be ready
    dist.barrier()
    if rank == 0:
        logger.info("\n" + "=" * 60)
        logger.info("🚀 ALL AGENTS READY - You can now launch workers!")
        logger.info("=" * 60 + "\n")

    try:
        agent.run()
    except KeyboardInterrupt:
        logger.info(f"\n[Agent {rank}] Interrupted by user")
    finally:
        agent.close()
        logger.info(f"[Agent {rank}] Cleanup complete. Exit.")


if __name__ == "__main__":
    main()
