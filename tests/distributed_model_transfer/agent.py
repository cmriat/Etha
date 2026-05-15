"""Minimal TensorBus agent for the distributed model transfer integration test."""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path

import posix_ipc
import torch.distributed as dist
from common import STORE_HOST, STORE_PORT, STORE_BACKEND, get_queue_state_paths

from etha.tensor_bus import TensorBusAgent


def _cleanup_stale(command_queue_path: str, state_path: str) -> None:
    for path_str in (command_queue_path, state_path):
        path = Path(path_str)
        for f in path.parent.glob(f"{path.name}*"):
            f.unlink(missing_ok=True)
        if "command" in path.name:
            stem = path.stem
            for sem in (f"/cq_{stem}", f"/cq_space_{stem}", f"/cq_ready_{stem}"):
                try:
                    posix_ipc.unlink_semaphore(sem)
                except posix_ipc.ExistentialError:
                    pass


def main() -> None:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])

    command_queue_path, state_path = get_queue_state_paths(rank)
    _cleanup_stale(command_queue_path, state_path)

    agent = TensorBusAgent(
        rank=rank,
        world_size=world_size,
        store_host=STORE_HOST,
        store_port=STORE_PORT,
        store_backend=STORE_BACKEND,
        lmdb_command_queue_path=command_queue_path,
        lmdb_state_path=state_path,
    )

    dist.barrier()
    try:
        agent.run()
    finally:
        agent.close()


if __name__ == "__main__":
    main()
