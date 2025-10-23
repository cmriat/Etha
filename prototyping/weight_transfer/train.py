"""Trainer for weight transfer."""

import os
import time
from multiprocessing.reduction import ForkingPickler

import torch
import posix_ipc
import torch.distributed as dist
from common import queue_path, store_tensor_payload

from etha.tensor_bus.commands import Ready, RegisterTensor
from etha.tensor_bus.command_queue import CommandQueue


class Trainer:
    def __init__(self, rank: int):
        self.rank = rank
        self.param = torch.tensor([rank], dtype=torch.float32, device="cuda")

    def forward_backward(self):
        time.sleep(10)

    def optimizer_step(self, step: int):
        self.param += 1.0
        if self.rank == 0:
            print(f"[train rank={self.rank}] step={step} value={self.param}")


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    queue_send = CommandQueue(queue_path(rank, "recv"))
    trainer = Trainer(rank)

    payload = ForkingPickler.dumps(trainer.param)
    tensor_id = f"weight_{rank}"
    store_tensor_payload(tensor_id, payload)
    queue_send.enqueue(
        RegisterTensor(tensor_id=tensor_id, storage_key=tensor_id, writer_pid=os.getpid(), timestamp=time.time())
    )

    finish_sem = posix_ipc.Semaphore(f"/weight_{rank}", flags=posix_ipc.O_CREAT, initial_value=0)

    try:
        for step in range(100):
            trainer.forward_backward()
            finish_sem.acquire()
            trainer.optimizer_step(step)
            queue_send.enqueue(Ready(tensor_id=tensor_id, timestamp=time.time()))

    finally:
        dist.destroy_process_group()
        finish_sem.close()
        try:
            finish_sem.unlink()
        except posix_ipc.ExistentialError:
            pass


if __name__ == "__main__":
    main()
