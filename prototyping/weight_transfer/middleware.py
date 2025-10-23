import os
import time
from multiprocessing.reduction import ForkingPickler

import torch
import torch.distributed as dist

from common import queue_path, load_tensor_payload
from etha.tensor_bus.command_queue import CommandQueue
from etha.tensor_bus.messages import Ready, Stop_Inference, FinishTransfer, RegisterTensor

def init_store(rank: int, world_size: int) -> dist.TCPStore:
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", 23456)
    store = dist.TCPStore(
        host_name=master_addr,
        port=int(master_port),
        world_size=world_size,
        is_master=(rank == 0),
    )
    return store

def transfer(rank: int, tensor: torch.Tensor, target_rank: int):
    time.sleep(2)
    if rank < 4:
        torch.distributed.send(tensor, target_rank)
    else:
        torch.distributed.recv(tensor, target_rank)
    if rank == 0:
        print(f"[middleware rank={rank}] tensor={tensor} transferred")

def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    is_train = rank < 4
    is_inference = rank >= 4
    target_rank = rank + 4 if is_train else rank - 4

    world_size = dist.get_world_size()
    store = init_store(rank, world_size)
    pstore = dist.PrefixStore("weight/", store)
    
    queue_send = CommandQueue(queue_path(rank, 'send'))
    queue_recv = CommandQueue(queue_path(rank, 'recv'))

    if is_train:
        tensor_id = f"weight_{rank}"
        queue_send.enqueue(FinishTransfer(tensor_id=tensor_id,timestamp=time.time()))

    tensors = {}
    while True:
        if queue_recv.size() == 0:
            if is_inference:
                if pstore.check([f"rank{rank}_stop"]):
                    pstore.delete_key(f"rank{rank}_stop")
                    queue_send.enqueue(Stop_Inference(timestamp=time.time()))
            else:
                time.sleep(0.005)
        else:
            command = queue_recv.dequeue()
            if isinstance(command, Ready):
                pstore.set(f"rank{rank}_ready", "1")
                if is_train:
                    pstore.set(f"rank{target_rank}_stop", "1")
                    pstore.wait([f"rank{target_rank}_ready"])
                    pstore.delete_key(f"rank{target_rank}_ready")
                transfer(rank, tensors[command.tensor_id], target_rank)
                queue_send.enqueue(FinishTransfer(tensor_id=command.tensor_id,timestamp=time.time()))
            elif isinstance(command, RegisterTensor):
                tensors[command.tensor_id] = ForkingPickler.loads(load_tensor_payload(command.tensor_id))
            else:
                raise ValueError(f"[middleware rank={rank}] unknown command {command}")
                break

    dist.destroy_process_group()
if __name__ == "__main__":
    main()