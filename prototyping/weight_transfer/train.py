import os
import time

from multiprocessing.reduction import ForkingPickler

import torch
import torch.distributed as dist
from common import queue_path, store_tensor_payload


from etha.tensor_bus.command_queue import CommandQueue
from etha.tensor_bus.messages import Ready, FinishTransfer, RegisterTensor



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

    queue_send = CommandQueue(queue_path(rank, 'recv'))
    queue_recv = CommandQueue(queue_path(rank, 'send'))
    trainer = Trainer(rank)

    payload = ForkingPickler.dumps(trainer.param)
    tensor_id = f"weight_{rank}"
    store_tensor_payload(tensor_id, payload)
    queue_send.enqueue(RegisterTensor(tensor_id=tensor_id,storage_key=tensor_id, writer_pid=os.getpid(),timestamp=time.time()))
    
    for step in range(100):
        trainer.forward_backward()

        while queue_recv.size() == 0:
            time.sleep(0.005)
        command = queue_recv.dequeue()
        assert isinstance(command, FinishTransfer)

        trainer.optimizer_step(step)

        queue_send.enqueue(Ready(tensor_id=tensor_id,timestamp=time.time()))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()