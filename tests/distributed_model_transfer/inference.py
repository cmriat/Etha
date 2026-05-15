"""Minimal inference engine for the distributed model transfer integration test."""

import os
import time

import torch
from common import PAIR_NAME, MESH_CONFIGS, EXPECTED_WORLD_SIZE, get_queue_state_paths, get_model_dtype_from_env
from transformers import AutoModelForCausalLM
from torch.distributed._tensor import DTensor, DeviceMesh, distribute_tensor

from etha.tensor_bus import bootstrap_client

PAIR_LOCAL = "distributed_inference"
PAIR_REMOTE = "distributed_training"
STRATEGY = os.environ.get("INFERENCE_STRATEGY", "hybrid_dp_mp")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
MODEL_DTYPE = get_model_dtype_from_env()


def _reset(module: torch.nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def _distribute(model: torch.nn.Module, mesh: DeviceMesh, placements: tuple) -> None:
    for name, param in list(model.named_parameters()):
        dp = torch.nn.Parameter(distribute_tensor(param, mesh, placements))
        parts = name.rsplit(".", 1)
        target = model.get_submodule(parts[0]) if len(parts) == 2 else model
        target._parameters[parts[-1]] = dp


def main() -> None:
    client, _ = bootstrap_client(path_naming_fn=get_queue_state_paths)
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)

    mesh_shape, placements = MESH_CONFIGS[STRATEGY]
    mesh = DeviceMesh("cuda", torch.arange(torch.prod(torch.tensor(mesh_shape))).view(mesh_shape))
    placements = tuple(placements)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=MODEL_DTYPE).to(device)
    model.apply(_reset)
    _distribute(model, mesh, placements)

    client.init_pair(
        pair_name=PAIR_NAME,
        local_name=PAIR_LOCAL,
        remote_name=PAIR_REMOTE,
        expected_world_size=EXPECTED_WORLD_SIZE,
        device_mesh=mesh,
        placements=placements,
        timeout=1000,
    )

    tensors = [(p.data.to_local(), PAIR_NAME) for _, p in model.named_parameters() if isinstance(p, DTensor)]

    for i in range(10):
        handler = client.register_tensors(batch_id=f"transfer_step_{i}", tensors=tensors, bucket_size=64 * 1024 * 1024)
        while not handler.query_transfer_signal():
            time.sleep(0.1)
        handler.transfer(transfer_type="recv", blocking=True, timeout=60)

    golden = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=MODEL_DTYPE).to(device)
    for name, param in model.named_parameters():
        if not isinstance(param, DTensor):
            continue
        assert torch.allclose(param.data.full_tensor(), golden.get_parameter(name))
    print("✅ Distributed model matches golden model")


if __name__ == "__main__":
    main()
