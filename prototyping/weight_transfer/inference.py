"""Inference engine for weight transfer."""

import os
import time
from multiprocessing.reduction import ForkingPickler

import torch
import posix_ipc
import torch.distributed as dist
from common import queue_path, store_tensor_payload

from etha.tensor_bus.commands import Ready, RegisterTensor, Stop_Inference
from etha.tensor_bus.command_queue import CommandQueue


class InferenceEngine:
    def __init__(self, rank: int):
        self.rank = rank
        self.param = torch.tensor([rank - 4], dtype=torch.float32, device="cuda")

    def step(self):
        time.sleep(2)
        if self.rank == 4:
            print(f"[inference rank={self.rank}] step value={self.param}")

    def stop(self):
        time.sleep(2)


def main():
    rank = int(os.environ["LOCAL_RANK"]) + 4
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    queue_send = CommandQueue(queue_path(rank, "recv"))
    queue_recv = CommandQueue(queue_path(rank, "send"))
    engine = InferenceEngine(rank)

    payload = ForkingPickler.dumps(engine.param)
    tensor_id = f"weight_{rank}"
    store_key = "recv_" + tensor_id
    store_tensor_payload(store_key, payload)
    queue_send.enqueue(
        RegisterTensor(tensor_id=tensor_id, storage_key=store_key, writer_pid=os.getpid(), timestamp=time.time())
    )

    finish_sem = posix_ipc.Semaphore(f"/weight_{rank}", flags=posix_ipc.O_CREAT, initial_value=0)

    try:
        while True:
            command = queue_recv.dequeue(block=False)
            if command is None:
                engine.step()
                continue

            if isinstance(command, Stop_Inference):
                engine.stop()
                queue_send.enqueue(Ready(tensor_id=tensor_id, timestamp=time.time()))
                finish_sem.acquire()
            else:
                raise ValueError(f"[inference rank={rank}] unknown command {command}")

    finally:
        dist.destroy_process_group()
        finish_sem.close()
        try:
            finish_sem.unlink()
        except posix_ipc.ExistentialError:
            pass


if __name__ == "__main__":
    main()
