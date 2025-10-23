"""Middleware for weight transfer."""

import os
import time
import threading
from multiprocessing.reduction import ForkingPickler

import torch
import posix_ipc
import torch.distributed as dist
from common import queue_path, load_tensor_payload

from etha.tensor_bus.commands import Ready, RegisterTensor, Stop_Inference
from etha.tensor_bus.command_queue import CommandQueue


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

    queue_send = CommandQueue(queue_path(rank, "send"))
    queue_recv = CommandQueue(queue_path(rank, "recv"))

    finish_sem = posix_ipc.Semaphore(f"/weight_{rank}", flags=posix_ipc.O_CREAT, initial_value=0)
    if is_train:
        finish_sem.release()

    tensors = {}
    stop_flag = threading.Event()

    def monitor_peer_ready():
        target_key = f"rank{target_rank}_ready"
        while True:
            if pstore.check([target_key]):
                if not stop_flag.is_set():
                    queue_send.enqueue(Stop_Inference(timestamp=time.time()))
                    stop_flag.set()
                time.sleep(0.1)
            else:
                time.sleep(0.1)

    monitor_thread = None
    if is_inference:
        monitor_thread = threading.Thread(target=monitor_peer_ready, daemon=True)
        monitor_thread.start()

    try:
        while True:
            command = queue_recv.dequeue(block=True)

            if isinstance(command, Ready):
                pstore.set(f"rank{rank}_ready", "1")
                pstore.wait([f"rank{target_rank}_ready"])
                pstore.delete_key(f"rank{target_rank}_ready")

                transfer(rank, tensors[command.tensor_id], target_rank)
                finish_sem.release()

                if is_inference:
                    stop_flag.clear()

            elif isinstance(command, RegisterTensor):
                tensors[command.tensor_id] = ForkingPickler.loads(load_tensor_payload(command.storage_key))
            else:
                raise ValueError(f"[middleware rank={rank}] unknown command {command}")

    finally:
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
        dist.destroy_process_group()
        finish_sem.close()
        try:
            finish_sem.unlink()
        except posix_ipc.ExistentialError:
            pass


if __name__ == "__main__":
    main()
