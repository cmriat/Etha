"""Distributed Worker with qwen model transfer."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/data/hf"
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import logging

import torch
import torch.nn as nn
from common import PAIR_NAME, MESH_CONFIGS, EXPECTED_WORLD_SIZE, get_queue_state_paths, get_model_dtype_from_env
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
LOCAL_NAME = "distributed_inference"
REMOTE_NAME = "distributed_training"

# Distributed strategy configuration
DISTRIBUTED_STRATEGY = os.environ.get("DISTRIBUTED_STRATEGY", "hybrid_dp_mp")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-30B-A3B")

MODEL_DTYPE = get_model_dtype_from_env()


def reset_parameters(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


class DistributedInferenceEngine:
    """Inference engine with distributed tensor support."""

    def __init__(self, rank: int, device: torch.device):
        self.rank = rank
        self.device = device
        # Setup device mesh based on strategy
        self.setup_device_mesh()

        # Create model
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=MODEL_DTYPE)
        logger.info(f"Rank {rank}: Model created from pretrained model")
        self.model.to(device)
        logger.info(f"Rank {rank}: Model moved to device")

        self.model.apply(reset_parameters)
        logger.info(f"Rank {rank}: Model parameters reset")
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

    torch.cuda.set_device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Distributed Inference Worker starting...")
    logger.info(f"  Global rank: {info.global_rank}")
    logger.info(f"  Agent rank: {info.agent_rank}")
    logger.info(f"  CUDA device: {device}")
    logger.info(f"  Distributed strategy: {DISTRIBUTED_STRATEGY}")
    logger.info(f"{'=' * 60}\n")

    # Create distributed inference engine
    engine = DistributedInferenceEngine(info.global_rank, device)

    # Register pair for distributed tensor transfer
    logger.info(f"Registering pair '{PAIR_NAME}' as '{LOCAL_NAME}' -> '{REMOTE_NAME}'")
    client.init_pair(
        pair_name=PAIR_NAME,
        local_name=LOCAL_NAME,
        remote_name=REMOTE_NAME,
        expected_world_size=EXPECTED_WORLD_SIZE,
        device_mesh=engine.device_mesh,
        placements=tuple(engine.placements),
        timeout=1000,
    )
    logger.info(f"✅ Pair '{PAIR_NAME}' registered successfully!")

    # Register the distributed tensors
    tensors_to_register = []
    for _, param in engine.model.named_parameters():
        if not isinstance(param, DTensor):  # Qwen-0.6B's lm_head
            continue
        tensors_to_register.append((param.data.to_local(), PAIR_NAME))

    # Transfer loop - register batch for each step
    i = 0
    while i < 10:
        batch_id = f"transfer_step_{i}"
        handler = client.register_tensors(batch_id=batch_id, tensors=tensors_to_register)
        logger.info(f"✅ Batch '{batch_id}' registered successfully!")

        # Wait for train side to signal it has sent data
        while not handler.query_transfer_signal():
            time.sleep(0.1)

        logger.info(f"step {i} transfer begin")
        handler.transfer(transfer_type="recv", blocking=True, timeout=60)
        logger.info(f"step {i} transfer completed")
        i += 1
    logger.info(f"✅ Received distributed model")

    golden_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=MODEL_DTYPE)
    golden_model.to(device)
    for name, param in engine.model.named_parameters():
        if not isinstance(param, DTensor):
            continue
        assert torch.allclose(param.data.full_tensor(), golden_model.get_parameter(name))
    logger.info(f"✅ Distributed model matches golden model")


if __name__ == "__main__":
    main()
