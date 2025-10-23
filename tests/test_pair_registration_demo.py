"""Simple demonstration of Pair registration.

This is a minimal test to verify:
1. Multiple Daemons can start and connect to TCPStore
2. RegisterPair messages can be sent and processed
3. Pair matching works (both sides discover each other)

Run this manually to test the basic flow before writing automated tests.
"""

import os
import sys
import time
import multiprocessing as mp

import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from etha.tensor_bus import PairHandler, TensorBusAgent


def run_daemon(rank: int, world_size: int, tcpstore_host: str, tcpstore_port: int):
    """Run a Daemon process."""
    # Set environment for torch.distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Create unique LMDB path for this rank
    lmdb_path = f"/tmp/tensor_bus_daemon_{rank}_command.lmdb"

    try:
        daemon = TensorBusAgent(
            rank=rank,
            world_size=world_size,
            tcpstore_host=tcpstore_host,
            tcpstore_port=tcpstore_port,
            lmdb_command_queue_path=lmdb_path,
        )

        print(f"[daemon-{rank}] Ready to process commands")

        # Run for a limited time (for demo purposes)
        start_time = time.time()
        timeout = 30  # Run for 30 seconds

        while time.time() - start_time < timeout:
            daemon.run()

    except Exception as e:
        print(f"[daemon-{rank}] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print(f"[daemon-{rank}] Exiting")


def run_host(rank: int, side_name: str, expected_world_size: int, lmdb_path: str, delay: float = 0):
    """Run a Host process that registers a pair."""
    time.sleep(delay)  # Stagger registration

    print(f"[host-{rank}] Starting registration...")

    # Create dummy tensor
    tensor = torch.zeros(10, dtype=torch.float32)

    try:
        handler = PairHandler(
            pair_name="test_pair",
            side_name=side_name,
            tensor=tensor,
            expected_side_world_size=expected_world_size,
            lmdb_command_queue_path=lmdb_path,
        )

        print(f"[host-{rank}] Registration complete!")

        # Keep alive for a bit
        time.sleep(5)

        handler.close()

    except Exception as e:
        print(f"[host-{rank}] Error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main demo: 2 Daemons, 2 sides, simple pair registration."""
    print("=" * 60)
    print("Tensor Bus Pair Registration Demo")
    print("=" * 60)
    print()
    print("Setup:")
    print("  - 2 Daemons (rank 0-1)")
    print("  - Side 'inference': rank 0")
    print("  - Side 'training': rank 1")
    print("  - Both register to pair 'test_pair'")
    print()
    print("=" * 60)
    print()

    world_size = 2
    tcpstore_host = "localhost"
    tcpstore_port = 29600

    # Spawn Daemon processes
    daemon_procs = []
    for rank in range(world_size):
        p = mp.Process(target=run_daemon, args=(rank, world_size, tcpstore_host, tcpstore_port))
        p.start()
        daemon_procs.append(p)
        time.sleep(0.5)  # Stagger startup

    print("[main] All Daemons started, waiting for them to initialize...")
    time.sleep(3)

    # Spawn Host processes
    host_procs = []

    # Inference side (rank 0)
    p0 = mp.Process(
        target=run_host,
        args=(0, "inference", 1, "/tmp/tensor_bus_daemon_0_command.lmdb", 0),
    )
    p0.start()
    host_procs.append(p0)

    # Training side (rank 1)
    p1 = mp.Process(
        target=run_host,
        args=(1, "training", 1, "/tmp/tensor_bus_daemon_1_command.lmdb", 1),
    )
    p1.start()
    host_procs.append(p1)

    print("[main] All Host processes started")

    # Wait for Host processes to complete
    for p in host_procs:
        p.join()

    print("[main] All Host processes completed")

    # Terminate Daemons
    for p in daemon_procs:
        p.terminate()
        p.join()

    print()
    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
