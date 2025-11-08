"""Distributed Training Worker with 4x4 tensor transfer for debugging."""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from common import PAIR_NAME, MESH_CONFIGS, TENSOR_SHAPE, get_queue_state_paths
from torch.distributed._tensor import DeviceMesh, distribute_tensor

from etha.tensor_bus import bootstrap_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Training Worker %(process)d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Parameters
LOCAL_NAME = "distributed_training"
REMOTE_NAME = "distributed_inference"
EXPECTED_WORLD_SIZE = 4

# Distributed strategy configuration
DISTRIBUTED_STRATEGY = os.environ.get("TRAINING_STRATEGY", "pure_mp")


class DistributedTrainer:
    """Training engine with distributed tensor support."""

    def __init__(self, rank: int, device: torch.device):
        self.rank = rank
        self.device = device

        # Create base tensor - 4x4 with specific pattern for easy debugging
        self.base_tensor = torch.arange(TENSOR_SHAPE.numel(), dtype=torch.float32, device=device).view(TENSOR_SHAPE)

        # Setup device mesh based on strategy
        self.setup_device_mesh()

        # Create distributed tensor
        self.distributed_param = distribute_tensor(self.base_tensor, self.device_mesh, tuple(self.placements))

        logger.info(f"Rank {rank}: Created distributed tensor with shape {self.distributed_param.shape}")
        logger.info(f"Rank {rank}: Local tensor shape: {self.distributed_param._local_tensor.shape}")

    def setup_device_mesh(self):
        """Setup device mesh configuration."""
        mesh_shape, self.placements = MESH_CONFIGS[DISTRIBUTED_STRATEGY]
        mesh_tensor = torch.arange(torch.prod(torch.tensor(mesh_shape))).view(mesh_shape)
        self.device_mesh = DeviceMesh("cuda", mesh_tensor)
        logger.info(f"Rank {self.rank}: Device mesh: {mesh_tensor}, placements: {self.placements}")

    def forward_backward(self):
        """Simulate forward and backward pass."""
        time.sleep(10)  # Simulate computation

    def optimizer_step(self, step: int):
        """Update parameters and log full distributed tensor."""
        self.distributed_param._local_tensor += 1
        full_tensor = self.distributed_param.full_tensor()

        if self.rank == 0:
            logger.info(f"[train rank={self.rank}] step={step} full_tensor=\n{full_tensor}")
        else:
            logger.debug(f"[train rank={self.rank}] step={step} full_tensor=\n{full_tensor}")


def main():
    # Bootstrap TensorBusClient
    client, info = bootstrap_client(path_naming_fn=get_queue_state_paths)
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Distributed Training Worker starting...")
    logger.info(f"  Global rank: {info.global_rank}")
    logger.info(f"  Agent rank: {info.agent_rank}")
    logger.info(f"  CUDA device: {device}")

    logger.info(f"  Distributed strategy: {DISTRIBUTED_STRATEGY}")
    logger.info(f"{'=' * 60}\n")

    # Create distributed trainer
    trainer = DistributedTrainer(info.global_rank, device)

    # Register pair for distributed tensor transfer
    logger.info(f"Registering pair '{PAIR_NAME}' as '{LOCAL_NAME}' -> '{REMOTE_NAME}'")
    handler = client.register_pair(
        pair_name=PAIR_NAME,
        local_name=LOCAL_NAME,
        remote_name=REMOTE_NAME,
        expected_world_size=EXPECTED_WORLD_SIZE,
        device_mesh=trainer.device_mesh,
        placements=tuple(trainer.placements),
    )

    # Register the distributed tensor
    sem = handler.register_tensor(
        tensor_name="distributed_param", tensor=trainer.distributed_param.to_local(), blocking=False
    )
    sem.acquire()
    sem.close()
    logger.info(f"✅ Pair '{PAIR_NAME}' registered successfully!")

    # Training loop with distributed weight transfer
    try:
        semaphore = None
        for step in range(50):
            trainer.forward_backward()
            if semaphore is not None:
                semaphore.acquire()
                semaphore.close()
            trainer.optimizer_step(step)

            # Send updated distributed tensor
            semaphore = handler.transfer(transfer_type="send", blocking=False)

            time.sleep(1)  # Simulate training iteration

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    finally:
        client.close()
        logger.info("Distributed training worker shutdown complete")


if __name__ == "__main__":
    main()
