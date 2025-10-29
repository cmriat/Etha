"""New Inference Worker for weight transfer using updated TensorBus architecture."""

import os
import time
import logging

import torch
from shared import PAIR_NAME, get_queue_state_paths

from etha.tensor_bus import bootstrap_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Inference Worker %(process)d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Parameters
LOCAL_NAME = "inference"
REMOTE_NAME = "training"
EXPECTED_WORLD_SIZE = 4  # Total inference workers


class InferenceEngine:
    def __init__(self, rank: int):
        self.rank = rank
        self.param = torch.tensor(0, dtype=torch.float32, device="cuda")

    def step(self):
        time.sleep(2)
        if self.rank == 0:
            logger.info(f"[inference rank={self.rank}] step value={self.param} device={self.param.device}")
        else:
            logger.debug(f"[inference rank={self.rank}] step value={self.param} device={self.param.device}")

    def resume(self):
        time.sleep(2)
        if self.rank == 0:
            logger.info(f"[inference rank={self.rank}] resume inference")
        else:
            logger.debug(f"[inference rank={self.rank}] resume inference")

    def stop(self):
        time.sleep(2)
        if self.rank == 0:
            logger.info(f"[inference rank={self.rank}] stop inference")
        else:
            logger.debug(f"[inference rank={self.rank}] stop inference")


def main():
    # Bootstrap TensorBusClient with rank offset for inference workers
    # Inference workers connect to agents 4-7, so we need AGENT_RANK_OFFSET=4
    os.environ["AGENT_RANK_OFFSET"] = "4"

    client, info = bootstrap_client(path_naming_fn=get_queue_state_paths)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Inference Worker starting...")
    logger.info(f"  Local rank: {info.local_rank}")
    logger.info(f"  Agent rank: {info.agent_rank}")
    logger.info(f"  CUDA device: {info.device}")
    logger.info(f"{'=' * 60}\n")

    # Create inference engine
    engine = InferenceEngine(info.local_rank)

    # Register pair for weight transfer
    logger.info(f"Registering pair '{PAIR_NAME}' as '{LOCAL_NAME}' -> '{REMOTE_NAME}'")
    handler = client.register_pair(
        pair_name=PAIR_NAME,
        local_name=LOCAL_NAME,
        remote_name=REMOTE_NAME,
        tensor=engine.param,
        expected_world_size=EXPECTED_WORLD_SIZE,
    )
    sem = handler.register_tensor(tensor_name="param", tensor=engine.param, blocking=False)
    sem.acquire()
    sem.close()
    logger.info(f"✅ Pair '{PAIR_NAME}' registered successfully!")

    # Inference loop with weight reception
    try:
        while True:
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
        logger.info("Inference worker shutdown complete")


if __name__ == "__main__":
    main()
