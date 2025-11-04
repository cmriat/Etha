"""Distributed Inference Worker with 4x4 tensor transfer for debugging."""

import os
import time
import logging

import torch
import torch.distributed as dist
from common import PAIR_NAME, TENSOR_SHAPE, read_placement, get_mesh_config, get_queue_state_paths
from torch.distributed._tensor import DeviceMesh, distribute_tensor

from etha.tensor_bus import bootstrap_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Inference Worker %(process)d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Parameters
LOCAL_NAME = "distributed_inference"
REMOTE_NAME = "distributed_training"
EXPECTED_WORLD_SIZE = 4

# Distributed strategy configuration
DISTRIBUTED_STRATEGY = os.environ.get("INFERENCE_STRATEGY", "hybrid_dp_mp")


class DistributedInferenceEngine:
    """Inference engine with distributed tensor support."""

    def __init__(self, rank: int, device: torch.device):
        self.rank = rank
        self.device = device

        # Create base tensor - 4x4 for easy debugging
        self.base_tensor = torch.randn(TENSOR_SHAPE, dtype=torch.float32, device=device)

        # Setup device mesh based on strategy
        self.setup_device_mesh()

        # Create distributed tensor
        self.distributed_param = distribute_tensor(self.base_tensor, self.device_mesh, tuple(self.placements))

        logger.info(f"Rank {rank}: Created distributed tensor with shape {self.distributed_param.shape}")
        logger.info(f"Rank {rank}: Local tensor shape: {self.distributed_param._local_tensor.shape}")

    def setup_device_mesh(self):
        """Setup device mesh configuration."""
        mesh_shape, placement_strs = get_mesh_config(DISTRIBUTED_STRATEGY)
        self.placements = read_placement(placement_strs)
        mesh_tensor = torch.arange(torch.prod(torch.tensor(mesh_shape))).view(mesh_shape)
        self.device_mesh = DeviceMesh(self.device, mesh_tensor)
        logger.info(f"Rank {self.rank}: Device mesh: {mesh_tensor}, placements: {self.placements}")

    def step(self):
        """Perform inference step."""
        time.sleep(2)  # Simulate inference work

        full_tensor = self.distributed_param.full_tensor()

        if self.rank == 0:
            logger.info(f"[inference rank={self.rank}] full_tensor=\n{full_tensor}")
        else:
            logger.debug(f"[inference rank={self.rank}] full_tensor=\n{full_tensor}")

    def resume(self):
        """Resume inference."""
        time.sleep(2)
        logger.info(f"[inference rank={self.rank}] resume inference")

    def stop(self):
        """Stop inference."""
        time.sleep(2)
        logger.info(f"[inference rank={self.rank}] stop inference")


def main():
    # Bootstrap TensorBusClient with rank offset for inference workers
    client, info = bootstrap_client(path_naming_fn=get_queue_state_paths)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Distributed Inference Worker starting...")
    logger.info(f"  Local rank: {info.local_rank}")
    logger.info(f"  Agent rank: {info.agent_rank}")
    logger.info(f"  CUDA device: {info.device}")
    logger.info(f"  Distributed strategy: {DISTRIBUTED_STRATEGY}")
    logger.info(f"{'=' * 60}\n")

    # Create distributed inference engine
    engine = DistributedInferenceEngine(info.local_rank, info.device)

    # Register pair for distributed tensor transfer
    logger.info(f"Registering pair '{PAIR_NAME}' as '{LOCAL_NAME}' -> '{REMOTE_NAME}'")
    handler = client.register_pair(
        pair_name=PAIR_NAME,
        local_name=LOCAL_NAME,
        remote_name=REMOTE_NAME,
        expected_world_size=EXPECTED_WORLD_SIZE,
        device_mesh=engine.device_mesh,
        placements=tuple(engine.placements),
    )

    # Register the distributed tensor
    sem = handler.register_tensor(
        tensor_name="distributed_param", tensor=engine.distributed_param.to_local(), blocking=False
    )
    sem.acquire()
    sem.close()
    logger.info(f"✅ Pair '{PAIR_NAME}' registered successfully!")

    # Inference loop with weight reception
    try:
        while True:
            dist.barrier()
            status = client.query_transfer_signal(PAIR_NAME)
            if status == True:
                engine.stop()
                handler.transfer(transfer_type="recv", blocking=True)
                engine.resume()
            engine.step()

    except KeyboardInterrupt:
        logger.info("\nInference interrupted by user")
    finally:
        client.close()
        logger.info("Distributed inference worker shutdown complete")


if __name__ == "__main__":
    main()
