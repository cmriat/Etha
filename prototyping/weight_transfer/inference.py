"""Inference engine for weight transfer."""

import os
import time
from multiprocessing.reduction import ForkingPickler

import torch
import torch.distributed as dist
from common import queue_path, store_tensor_payload

from etha.tensor_bus.messages import Ready, FinishTransfer, RegisterTensor, Stop_Inference
from etha.tensor_bus.command_queue import CommandQueue


class InferenceEngine:
    def __init__(self, rank: int):
        self.rank = rank
        self.param = torch.tensor([rank - 4], dtype=torch.float32, device="cuda")

    def step(self):
        time.sleep(2.0)
        if self.rank == 4:
            print(f"[inference rank={self.rank}] step value={self.param}")


def main():
    rank = int(os.environ["LOCAL_RANK"]) + 4
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    queue_send = CommandQueue(queue_path(rank, "recv"))
    queue_recv = CommandQueue(queue_path(rank, "send"))
    engine = InferenceEngine(rank)

    payload = ForkingPickler.dumps(engine.param)
    tensor_id = f"weight_recv_{rank}"
    store_tensor_payload(tensor_id, payload)
    queue_send.enqueue(
        RegisterTensor(tensor_id=tensor_id, storage_key=tensor_id, writer_pid=os.getpid(), timestamp=time.time())
    )

    while True:
        if queue_recv.size() == 0:
            engine.step()
        else:
            command = queue_recv.dequeue()
            if isinstance(command, Stop_Inference):
                time.sleep(2.0)
                queue_send.enqueue(Ready(tensor_id=tensor_id, timestamp=time.time()))
            elif isinstance(command, FinishTransfer):
                time.sleep(2.0)
            else:
                raise ValueError(f"[inference rank={rank}] unknown command {command}")
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
