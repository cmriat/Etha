"""Training Worker Process - Pair Registration Demo.

Launch with torchrun:
    torchrun --nproc_per_node=4 prototyping/pair_registration_demo/worker_training.py

This will start 4 real worker processes that:
1. Each worker connects to daemon_rank{local_rank+4} (4-7)
2. Register pair "obs" with side_name="training"
3. Poll Daemon's state LMDB until pair is matched
4. Print matched result
"""

import os
import sys
import time
import logging

# Ensure we can import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# CUDA allocator config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from shared import PAIR_NAME, get_agent_state_path, get_agent_command_queue_path

from etha.tensor_bus import TensorBusClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Training Worker %(process)d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Parameters
LOCAL_NAME = "training"
REMOTE_NAME = "inference"
EXPECTED_WORLD_SIZE = 4  # Total training workers
AGENT_RANK_OFFSET = 4  # Training workers map to agent ranks 4-7


def main():
    # Get rank from environment (torchrun sets LOCAL_RANK)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    agent_rank = local_rank + AGENT_RANK_OFFSET  # Training workers map to agent ranks 4-7
    device = f"cuda:{local_rank + AGENT_RANK_OFFSET}"

    print(f"\n{'=' * 60}")
    print(f"Training Worker (local_rank={local_rank}) starting...")
    print(f"Agent rank: {agent_rank}")
    print(f"Device: {device}")
    print(f"{'=' * 60}\n")

    # Set CUDA device
    torch.cuda.set_device(local_rank + AGENT_RANK_OFFSET)

    # Create dummy tensor
    tensor = torch.zeros(10, dtype=torch.float32, device=device)
    logger.info(f"Worker {local_rank}: Created tensor on {device}")

    # Get Agent paths
    command_queue_path = get_agent_command_queue_path(agent_rank)
    state_path = get_agent_state_path(agent_rank)

    logger.info(f"Worker {local_rank}: Connecting to Agent {agent_rank}")
    logger.info(f"  CommandQueue: {command_queue_path}")
    logger.info(f"  State LMDB: {state_path}")

    # Create TensorBusClient
    client = TensorBusClient(
        lmdb_command_queue_path=command_queue_path,
        agent_state_lmdb_path=state_path,
    )

    # Register pair (blocks until matched)
    logger.info(f"Worker {local_rank}: Registering pair '{PAIR_NAME}' as '{LOCAL_NAME}' -> '{REMOTE_NAME}'")

    start_time = time.time()
    handler = client.register_pair(
        pair_name=PAIR_NAME,
        local_name=LOCAL_NAME,
        remote_name=REMOTE_NAME,
        tensor=tensor,
        expected_world_size=EXPECTED_WORLD_SIZE,
    )
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"✅ Worker {local_rank}: Pair '{PAIR_NAME}' matched!")
    print(f"   Elapsed time: {elapsed:.2f}s")
    print(f"{'=' * 60}\n")

    # Keep alive for a bit (optional, for debugging)
    logger.info(f"Worker {local_rank}: Pair registered, keeping alive for 5s...")
    time.sleep(5)

    # Cleanup
    client.close()
    logger.info(f"Worker {local_rank}: Exit")


if __name__ == "__main__":
    main()
