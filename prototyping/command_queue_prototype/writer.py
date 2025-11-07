"""Writer Process - CommandQueue Prototype.

Creates a CUDA tensor, registers it via CommandQueue, then continuously modifies it.
The reader process should see these modifications in real-time (zero-copy).
"""

import os
import sys
import time
import ctypes

# Ensure we can import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from multiprocessing.reduction import ForkingPickler

import torch
from shared import LMDB_QUEUE_PATH

from etha.tensor_bus import CommandQueue, RegisterTensorBatch

# Constants for ptrace authorization
PR_SET_PTRACER = 0x59616D61
PR_SET_PTRACER_ANY = -1
libc = ctypes.CDLL("libc.so.6", use_errno=True)


def setup_ptrace():
    """Setup ptrace authorization for reader process.

    For simplicity, we use PR_SET_PTRACER_ANY in this prototype.
    In production, you'd authorize specific reader PIDs.
    """
    result = libc.prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    if result == 0:
        print("[writer] ✅ Ptrace authorized (ANY)")
    else:
        print(f"[writer] ⚠️  Ptrace setup failed: {result}")


def main():
    device = "cuda:0"
    torch.cuda.set_device(0)

    print(f"[writer] PID={os.getpid()}")
    print(f"[writer] Device: {torch.cuda.get_device_name(0)}")

    # Setup ptrace
    setup_ptrace()

    # Create CUDA tensor
    tensor_id = "tensor_0"
    t = torch.zeros(10, dtype=torch.float32, device=device).contiguous()

    print(f"\n[writer] Created tensor '{tensor_id}':")
    print(f"  Shape: {t.shape}")
    print(f"  Dtype: {t.dtype}")
    print(f"  Device: {t.device}")
    print(f"  CUDA ptr: {t.data_ptr():#x}")
    print(f"  Initial values: {t.cpu().numpy()}")

    # Store tensor payload in LMDB storage
    payload = ForkingPickler.dumps(t)

    # Send RegisterTensorBatch command via CommandQueue
    queue = CommandQueue(LMDB_QUEUE_PATH)
    msg = RegisterTensorBatch(
        pair_name="pair_0", tensor_names=[tensor_id], tensor_payloads=[payload], timestamp=time.time()
    )
    queue.enqueue(msg)
    queue.close()

    print(f"[writer] ✅ Sent RegisterTensorBatch via CommandQueue")
    print(f"  Message fields: pair_name, tensor_name, tensor_payload, timestamp")
    print(f"  Tensor metadata (shape, dtype, device, ptr) in pickled payload")
    print(f"\n[writer] Now start reader.py in another terminal:")
    print(f"  pixi run -e dev python prototyping/command_queue_prototype/reader.py")

    # Wait for reader to start
    # input("\n[writer] Press Enter after reader starts...")

    # Continuously modify tensor values
    print("\n[writer] Starting tensor modification loop...")
    print("[writer] Reader should see these changes in real-time!\n")

    try:
        for i in range(100):
            # Fill tensor with value i
            t.fill_(float(i))
            torch.cuda.synchronize()

            print(f"[writer] Iteration {i:3d}: set all values to {i}")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n[writer] Interrupted by user")

    print("[writer] Done. Keeping tensor alive...")
    print("[writer] Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[writer] Cleaning up...")

        # Cleanup: destroy command queue
        cleanup_queue = CommandQueue(LMDB_QUEUE_PATH)
        cleanup_queue.destroy()
        print("[writer] ✅ CommandQueue destroyed")

        print("[writer] Exit")


if __name__ == "__main__":
    main()
