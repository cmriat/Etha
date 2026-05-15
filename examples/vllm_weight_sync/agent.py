"""Etha agent: holds the NCCL process group and drives send/recv on behalf of workers."""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import signal
import logging
from pathlib import Path

import posix_ipc
import torch.distributed as dist
from common import CONFIG, get_queue_state_paths

from etha.tensor_bus import TensorBusAgent

signal.signal(signal.SIGTERM, signal.default_int_handler)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [agent rk%(process)d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _cleanup_stale(rank: int, command_queue_path: str, state_path: str) -> None:
    """Remove old LMDB files and POSIX semaphores left from a prior run."""
    for path_str in (command_queue_path, state_path):
        path = Path(path_str)
        for f in path.parent.glob(f"{path.name}*"):
            f.unlink(missing_ok=True)
            logger.debug("rank %d cleaned up LMDB file %s", rank, f)

        if "command" in path.name:
            stem = path.stem
            for sem in (f"/cq_{stem}", f"/cq_space_{stem}", f"/cq_ready_{stem}"):
                try:
                    posix_ipc.unlink_semaphore(sem)
                    logger.debug("rank %d cleaned up semaphore %s", rank, sem)
                except posix_ipc.ExistentialError:
                    pass


def main() -> None:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])

    logger.info("rank %d starting, world_size=%d, store=%s:%s", rank, world_size, CONFIG.store_host, CONFIG.store_port)

    command_queue_path, state_path = get_queue_state_paths(rank)
    _cleanup_stale(rank, command_queue_path, state_path)
    logger.info("rank %d command_queue=%s state=%s", rank, command_queue_path, state_path)

    agent = TensorBusAgent(
        rank=rank,
        world_size=world_size,
        store_host=CONFIG.store_host,
        store_port=CONFIG.store_port,
        store_backend=CONFIG.store_backend,
        lmdb_command_queue_path=command_queue_path,
        lmdb_state_path=state_path,
    )
    logger.info("rank %d initialized, entering main loop", rank)

    dist.barrier()
    if rank == 0:
        logger.info("all %d agents ready", world_size)

    try:
        agent.run()
    except KeyboardInterrupt:
        logger.info("rank %d interrupted", rank)
    finally:
        agent.close()
        logger.info("rank %d shutdown complete", rank)


if __name__ == "__main__":
    main()
