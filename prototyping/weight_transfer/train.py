"""New Training Worker for weight transfer using updated TensorBus architecture."""

import time
import logging

import torch
from shared import PAIR_NAME, get_queue_state_paths

from etha.tensor_bus import bootstrap_client

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


class Trainer:
    def __init__(self, rank: int):
        self.rank = rank
        self.param = torch.tensor(0, dtype=torch.float32, device="cuda")

    def forward_backward(self):
        time.sleep(2)  # Simulate training work

    def optimizer_step(self, step: int):
        self.param += 1.0
        if self.rank == 0:
            logger.info(f"[train rank={self.rank}] step={step} value={self.param} device={self.param.device}")
        else:
            logger.debug(f"[train rank={self.rank}] step={step} value={self.param} device={self.param.device}")


def main():
    # Bootstrap TensorBusClient
    client, info = bootstrap_client(path_naming_fn=get_queue_state_paths)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training Worker starting...")
    logger.info(f"  Local rank: {info.local_rank}")
    logger.info(f"  Agent rank: {info.agent_rank}")
    logger.info(f"  CUDA device: {info.device}")
    logger.info(f"{'=' * 60}\n")

    # Create trainer
    trainer = Trainer(info.local_rank)

    # Register pair for weight transfer
    logger.info(f"Registering pair '{PAIR_NAME}' as '{LOCAL_NAME}' -> '{REMOTE_NAME}'")
    handler = client.register_pair(
        pair_name=PAIR_NAME,
        local_name=LOCAL_NAME,
        remote_name=REMOTE_NAME,
        expected_world_size=EXPECTED_WORLD_SIZE,
    )
    sem = handler.register_tensor(tensor_name="param", tensor=trainer.param, blocking=False)
    sem.acquire()
    sem.close()
    logger.info(f"✅ Pair '{PAIR_NAME}' registered successfully!")

    # Training loop with weight transfer
    try:
        semaphore = None
        for step in range(50):
            trainer.forward_backward()
            if semaphore is not None:
                semaphore.acquire()
                semaphore.close()
            trainer.optimizer_step(step)
            semaphore = handler.transfer(transfer_type="send")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    finally:
        client.close()
        logger.info("Training worker shutdown complete")


if __name__ == "__main__":
    main()
