"""Ray-based Distributed Inference Worker for NCCL device ID debugging.

This script uses Ray actors instead of torchrun to launch inference workers,
allowing us to reproduce CUDA device ID mapping issues when Ray virtualizes GPUs.
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray
import torch

# Note: common module and etha.tensor_bus imports moved to InferenceActor.__init__ to avoid Ray worker import errors
from torch.distributed._tensor import DeviceMesh, distribute_tensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Ray Inference Worker %(process)d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Note: PAIR_NAME, TENSOR_SHAPE, LOCAL_NAME, REMOTE_NAME, EXPECTED_WORLD_SIZE
# are now loaded inside InferenceActor.__init__ from common module


class DistributedInferenceEngine:
    """Inference engine with distributed tensor support."""

    def __init__(self, rank: int, device: torch.device, strategy: str, tensor_shape: torch.Size, MESH_CONFIGS: dict):
        self.rank = rank
        self.device = device

        # Create base tensor - 4x4 for easy debugging
        self.base_tensor = torch.randn(tensor_shape, dtype=torch.float32, device=device)

        # Setup device mesh based on strategy
        self.MESH_CONFIGS = MESH_CONFIGS

        self.setup_device_mesh(strategy)

        # Create distributed tensor
        self.distributed_param = distribute_tensor(self.base_tensor, self.device_mesh, tuple(self.placements))

        logger.info(f"Rank {rank}: Created distributed tensor with shape {self.distributed_param.shape}")
        logger.info(f"Rank {rank}: Local tensor shape: {self.distributed_param._local_tensor.shape}")

    def setup_device_mesh(self, strategy: str):
        """Setup device mesh configuration."""
        mesh_shape, self.placements = self.MESH_CONFIGS[strategy]
        mesh_tensor = torch.arange(torch.prod(torch.tensor(mesh_shape))).view(mesh_shape)
        self.device_mesh = DeviceMesh("cuda", mesh_tensor)
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


@ray.remote(num_gpus=1)
class InferenceActor:
    """Ray Actor for distributed inference worker.

    Manually injects environment variables to simulate torchrun's SPMD behavior.
    """

    def __init__(self, rank: int, world_size: int, master_port: int, agent_rank_offset: int, strategy: str):
        """Initialize Ray inference actor.

        Args:
            rank: Process rank (0-3)
            world_size: Total number of processes (4)
            master_port: Master port for NCCL rendezvous (39502)
            agent_rank_offset: Offset to map to agent ranks (4)
            strategy: Distributed strategy (hybrid_dp_mp or pure_mp)
        """
        # IMPORTANT: Ray actors run in separate worker processes
        # Need to set up sys.path again for imports to work
        import os
        import sys

        # Get the absolute path to the prototyping directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prototyping_dir = os.path.dirname(script_dir)
        if prototyping_dir not in sys.path:
            sys.path.insert(0, prototyping_dir)

        # Import common module here to ensure it's available in Ray worker
        from common import PAIR_NAME, MESH_CONFIGS, TENSOR_SHAPE, get_queue_state_paths

        from etha.tensor_bus import bootstrap_client

        # Store imports as instance variables for later use
        self.PAIR_NAME = PAIR_NAME
        self.TENSOR_SHAPE = TENSOR_SHAPE
        self.MESH_CONFIGS = MESH_CONFIGS
        self.get_queue_state_paths = get_queue_state_paths
        self.bootstrap_client = bootstrap_client

        # Manually inject environment variables (simulate torchrun)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["AGENT_RANK_OFFSET"] = str(agent_rank_offset)
        os.environ["INFERENCE_STRATEGY"] = strategy

        # Print diagnostic information
        self._print_env_debug()

        # Bootstrap TensorBusClient (will call dist.init_process_group internally)
        self.client, self.info = self.bootstrap_client(path_naming_fn=self.get_queue_state_paths)

        # Manually set CUDA device (bootstrap no longer does this)
        torch.cuda.set_device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Ray Inference Actor initialized")
        logger.info(f"  Global rank: {self.info.global_rank}")
        logger.info(f"  Agent rank: {self.info.agent_rank}")
        logger.info(f"  CUDA device: {device}")
        logger.info(f"  Distributed strategy: {strategy}")
        logger.info(f"{'=' * 60}\n")

        # Constants for pair registration
        LOCAL_NAME = "distributed_inference"
        REMOTE_NAME = "distributed_training"
        EXPECTED_WORLD_SIZE = 4

        # Create distributed inference engine
        self.engine = DistributedInferenceEngine(
            self.info.global_rank, device, strategy, self.TENSOR_SHAPE, self.MESH_CONFIGS
        )

        # Register pair for distributed tensor transfer
        logger.info(f"Registering pair '{self.PAIR_NAME}' as '{LOCAL_NAME}' -> '{REMOTE_NAME}'")
        self.client.init_pair(
            pair_name=self.PAIR_NAME,
            local_name=LOCAL_NAME,
            remote_name=REMOTE_NAME,
            expected_world_size=EXPECTED_WORLD_SIZE,
            device_mesh=self.engine.device_mesh,
            placements=tuple(self.engine.placements),
        )

        logger.info(f"✅ Pair '{self.PAIR_NAME}' registered successfully!")

    def _print_env_debug(self):
        """Print diagnostic information about environment and CUDA devices."""
        logger.info("\n" + "=" * 80)
        logger.info(f"RAY ACTOR DEBUG INFO [PID={os.getpid()}]")
        logger.info("=" * 80)
        logger.info(f"Environment Variables:")
        logger.info(f"  RANK: {os.environ.get('RANK', 'NOT SET')}")
        logger.info(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'NOT SET')}")
        logger.info(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")
        logger.info(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT SET')}")
        logger.info(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT SET')}")
        logger.info(f"  AGENT_RANK_OFFSET: {os.environ.get('AGENT_RANK_OFFSET', 'NOT SET')}")
        logger.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")

        logger.info(f"\nCUDA Information:")
        logger.info(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        logger.info(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")

        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            logger.info(f"  torch.cuda.current_device(): {current_device}")
            logger.info(f"  torch.cuda.get_device_name({current_device}): {torch.cuda.get_device_name(current_device)}")

            # Get device properties
            props = torch.cuda.get_device_properties(current_device)
            logger.info(f"  Device total memory: {props.total_memory / 1e9:.2f} GB")
            logger.info(f"  GPU UUID: {props.uuid}")

        logger.info("=" * 80 + "\n")

    def run(self):
        """Run the inference loop.

        Synchronizes with train side using matching batch_id pattern.
        Each step waits for train side to send data before receiving.
        """
        logger.info(f"Starting inference loop for rank {self.info.global_rank}...")

        try:
            for step in range(50):
                # Register tensors with unique batch_id matching train side
                # MUST register before waiting for signal to avoid deadlock
                batch_id = f"transfer_step_{step}"
                handler = self.client.register_tensors(
                    batch_id=batch_id, tensors=[(self.engine.distributed_param.to_local(), self.PAIR_NAME)]
                )

                # Wait for train side to signal it has sent data for this step
                while not handler.query_transfer_signal():
                    time.sleep(0.1)

                self.engine.stop()

                # Receive updated distributed tensor
                handler.transfer(transfer_type="recv", blocking=True)

                self.engine.resume()
                self.engine.step()

        except KeyboardInterrupt:
            logger.info("\nInference interrupted by user")
        except Exception as e:
            logger.error(f"Error in inference loop: {e}")
            import traceback

            traceback.print_exc()
            raise
        finally:
            self.client.close()
            logger.info(f"Ray inference actor (rank {self.info.global_rank}) shutdown complete")


def main():
    """Launch Ray-based inference workers (SPMD with manual env injection)."""
    # Configuration
    world_size = 4
    master_port = 39502
    agent_rank_offset = 4
    strategy = os.environ.get("DISTRIBUTED_STRATEGY", "hybrid_dp_mp")
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

    print("=" * 80)
    print("RAY-BASED DISTRIBUTED INFERENCE LAUNCHER")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  World size: {world_size}")
    print(f"  Master port: {master_port}")
    print(f"  Agent rank offset: {agent_rank_offset}")
    print(f"  Distributed strategy: {strategy}")
    print("=" * 80 + "\n")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Create actors in SPMD fashion (for loop simulates torchrun)
        print(f"Creating {world_size} Ray inference actors...")
        actors = []
        for rank in range(world_size):
            print(f"  Launching actor for rank {rank}...")
            actor = InferenceActor.remote(
                rank=rank,
                world_size=world_size,
                master_port=master_port,
                agent_rank_offset=agent_rank_offset,
                strategy=strategy,
            )
            actors.append(actor)

        print(f"\n✅ All {world_size} actors created successfully!")
        print("\nStarting inference loops...")

        # Start all actors' run loops (blocking)
        futures = [actor.run.remote() for actor in actors]
        ray.get(futures)

    except KeyboardInterrupt:
        print("\n\nLauncher interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in launcher: {e}")
        import traceback

        traceback.print_exc()
    finally:
        ray.shutdown()
        print("\nRay shutdown complete")


if __name__ == "__main__":
    main()
