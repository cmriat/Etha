"""Reader Process - CommandQueue Prototype.

Dequeues RegisterTensor message, rebuilds the tensor, then monitors value changes.
If zero-copy works, it should see the writer's modifications in real-time.
"""

import os
import sys
import time

# Ensure we can import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from multiprocessing.reduction import ForkingPickler

import torch
from shared import LMDB_QUEUE_PATH, load_tensor_payload

from etha.tensor_bus import CommandQueue, RegisterTensor


def main():
    torch.cuda.set_device(0)

    print(f"[reader] PID={os.getpid()}")
    print(f"[reader] Device: {torch.cuda.get_device_name(0)}")

    # Dequeue RegisterTensor message from CommandQueue
    print(f"\n[reader] Waiting for RegisterTensor message...")
    queue = CommandQueue(LMDB_QUEUE_PATH)
    msg = queue.dequeue()
    queue.close()

    if msg is None:
        print("[reader] ❌ No messages in queue!")
        print("[reader] Make sure writer.py is running first.")
        return

    if not isinstance(msg, RegisterTensor):
        print(f"[reader] ❌ Unexpected message type: {type(msg)}")
        print(f"[reader] Expected RegisterTensor, got {msg}")
        return

    # Display received message
    print(f"\n[reader] ✅ Received RegisterTensor message:")
    print(f"  tensor_id: {msg.tensor_id}")
    print(f"  storage_key: {msg.storage_key}")
    print(f"  writer_pid: {msg.writer_pid}")
    print(f"  timestamp: {msg.timestamp}")

    # Load tensor payload from LMDB storage
    print(f"\n[reader] Loading tensor payload from LMDB...")
    payload = load_tensor_payload(msg.storage_key)

    if payload is None:
        print(f"[reader] ❌ Tensor payload not found for key '{msg.storage_key}'")
        return

    # Rebuild tensor (zero-copy via ForkingPickler)
    t = ForkingPickler.loads(payload)

    print(f"\n[reader] ✅ Rebuilt tensor from pickled payload:")
    print(f"  Shape: {t.shape}")
    print(f"  Dtype: {t.dtype}")
    print(f"  Device: {t.device}")
    print(f"  CUDA ptr: {t.data_ptr():#x}")
    print(f"\n[reader] Note: All metadata came from ForkingPickler, not message!")

    print(f"\n[reader] Initial values: {t.cpu().numpy()}")

    # Monitor tensor value changes
    print(f"\n[reader] Monitoring tensor values...")
    print(f"[reader] If zero-copy works, you should see the writer's modifications!")
    print(f"[reader] Press Ctrl+C to stop\n")

    try:
        iteration = 0
        while True:
            torch.cuda.synchronize()  # Ensure we see latest GPU data
            values = t.cpu().numpy()

            print(f"[reader] Iteration {iteration:3d}: {values}")

            iteration += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[reader] Interrupted by user")

    print("[reader] Exit")


if __name__ == "__main__":
    main()
