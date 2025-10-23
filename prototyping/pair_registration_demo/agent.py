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
import sys
import logging
from pathlib import Path

# Ensure we can import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# CUDA allocator config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    level=logging.DEBUG,  # Changed to DEBUG for detailed logs
    format="[%(asctime)s] [Agent %(process)d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    # Get rank and world_size from environment (torchrun sets these)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", AGENT_WORLD_SIZE))

    print(f"\n{'=' * 60}")
    print(f"Agent Rank {rank} starting...")
    print(f"World Size: {world_size}")
    print(f"TCPStore: {TCPSTORE_HOST}:{TCPSTORE_PORT}")
    print(f"{'=' * 60}\n")

    # Get LMDB paths
    command_queue_path = get_agent_command_queue_path(rank)
    state_path = get_agent_state_path(rank)

    # Clean up old LMDB files (remove stale data and locks)
    for path_str in [command_queue_path, state_path]:
        path = Path(path_str)
        for f in path.parent.glob(f"{path.name}*"):
            f.unlink(missing_ok=True)
            print(f"[Agent {rank}] Cleaned up: {f}")

    print(f"[Agent {rank}] CommandQueue: {command_queue_path}")
    print(f"[Agent {rank}] State LMDB: {state_path}\n")

    # Initialize Agent
    agent = TensorBusAgent(
        rank=rank,
        world_size=world_size,
        tcpstore_host=TCPSTORE_HOST,
        tcpstore_port=TCPSTORE_PORT,
        lmdb_command_queue_path=command_queue_path,
        lmdb_state_path=state_path,
    )

    print(f"[Agent {rank}] ✅ Initialized successfully")
    print(f"[Agent {rank}] Entering main loop (polling for commands)...\n")

    # Wait for all agents to be ready
    dist.barrier()
    if rank == 0:
        print("\n" + "=" * 60)
        print("🚀 ALL AGENTS READY - You can now launch workers!")
        print("=" * 60 + "\n")

    try:
        agent.run()
    except KeyboardInterrupt:
        print(f"\n[Agent {rank}] Interrupted by user")
    finally:
        agent.close()
        print(f"[Agent {rank}] Cleanup complete. Exit.")


if __name__ == "__main__":
    main()
