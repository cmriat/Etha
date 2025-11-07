"""Distributed Worker with qwen model transfer."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/data/hf"
os.environ["HF_HUB_OFFLINE"] = "1"

import logging

import torch
import torch.nn as nn
from common import PAIR_NAME, MESH_CONFIGS, EXPECTED_WORLD_SIZE, get_queue_state_paths
from transformers import AutoModelForCausalLM
from torch.distributed._tensor import DTensor, DeviceMesh, distribute_tensor

from etha.tensor_bus import bootstrap_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Worker %(process)d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Parameters
LOCAL_NAME = "distributed_training"
REMOTE_NAME = "distributed_inference"

# Distributed strategy configuration
DISTRIBUTED_STRATEGY = os.environ.get("TRAINING_STRATEGY", "pure_mp")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-30B-A3B")


class DistributedTrainer:
    """Training engine with distributed tensor support."""

    def __init__(self, rank: int, device: torch.device):
        self.rank = rank
        self.device = device
        # Setup device mesh based on strategy
        self.setup_device_mesh()

        # Create model
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        logger.info(f"Rank {rank}: Model created from pretrained model")
        self.model.to(device)
        logger.info(f"Rank {rank}: Model moved to device")

        self.distribute_model()

        logger.info(f"Rank {rank}: Distributed model")

    def setup_device_mesh(self):
        """Setup device mesh configuration."""
        mesh_shape, self.placements = MESH_CONFIGS[DISTRIBUTED_STRATEGY]
        mesh_tensor = torch.arange(torch.prod(torch.tensor(mesh_shape))).view(mesh_shape)
        self.device_mesh = DeviceMesh("cuda", mesh_tensor)
        logger.info(f"Rank {self.rank}: Device mesh: {mesh_tensor}, placements: {self.placements}")

    def distribute_model(self):
        """Distribute model."""
        for name, param in self.model.named_parameters():
            dist_param = nn.Parameter(distribute_tensor(param, self.device_mesh, tuple(self.placements)))
            self._assign_parameter(name, dist_param)

    def _assign_parameter(self, name: str, parameter: nn.Parameter) -> None:
        """Replace an existing Parameter, handling nested module names with dots."""
        if "." in name:
            module_name, param_name = name.rsplit(".", 1)
            target_module = self.model.get_submodule(module_name)
        else:
            target_module = self.model
            param_name = name
        target_module._parameters[param_name] = parameter


def main():
    # Bootstrap TensorBusClient
    client, info = bootstrap_client(path_naming_fn=get_queue_state_paths)

    try:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Distributed Training Worker starting...")
        logger.info(f"  Global rank: {info.global_rank}")
        logger.info(f"  Agent rank: {info.agent_rank}")
        logger.info(f"  CUDA device: {info.device}")
        logger.info(f"  Distributed strategy: {DISTRIBUTED_STRATEGY}")
        logger.info(f"{'=' * 60}\n")

        # Create distributed trainer
        trainer = DistributedTrainer(info.global_rank, info.device)

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
        logger.info(f"✅ Pair '{PAIR_NAME}' registered successfully!")
        # Register the distributed tensor
        tensor_names = []
        tensor_data = []
        for name, param in trainer.model.named_parameters():
            if not isinstance(param, DTensor):
                continue
            tensor_names.append(name)
            tensor_data.append(param.data.to_local())

        # Batch register all tensors
        sem = handler.register_tensor_batch(
            tensor_names=tensor_names,
            tensors=tensor_data,
            blocking=False,
        )
        sem.acquire()
        sem.close()

        logger.info(f"✅Tensors for Pair '{PAIR_NAME}' registered successfully!")

        handler.transfer(transfer_type="send", blocking=True, timeout=60)
        logger.info(f"✅ Sent distributed model")
    finally:
        client.close()
        logger.info("Distributed training worker shutdown complete")


if __name__ == "__main__":
    main()
