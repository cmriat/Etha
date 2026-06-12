"""FSDP2 trainer:torchrun 起 N 卡,加载真模型 fully_shard,提供 collective RPC。

driver 经 rpc.CollectiveClient 调下面四个方法,与 vLLM 侧的 worker extension
对称(README 时序)。
"""

import os

import torch
import torch.distributed as dist
from rpc import serve
from protocol import build_chunks
from fsdp_side import FsdpWeightProtocol
from transformers import AutoModelForCausalLM
from torch.distributed.fsdp import fully_shard

from etha import chunk_comm, create_cross_group

RPC_PORT_BASE = 52100


class TrainerWorker:
    def __init__(self):
        model = AutoModelForCausalLM.from_pretrained(
            os.environ.get("ETHA_MODEL", "Qwen/Qwen3-0.6B"), dtype=torch.bfloat16
        ).cuda()
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)
        self.api = FsdpWeightProtocol(model, base_rank=0)
        self.rank = dist.get_rank()

    def manifest(self):
        return {n: (tuple(p.shape), p.dtype) for n, p in self.api._params.items()}

    def etha_export(self):
        return {n: self.api.get_sharding(n) for n in self.api._params}

    def etha_init(self, host, port, world, names, peer):
        self.group = create_cross_group(host, port, self.rank, world)
        self.chunks = build_chunks(self.api, names, peer, self.rank, sending=True)

    def etha_transfer(self):
        chunk_comm(self.chunks, group=self.group)


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    worker = TrainerWorker()
    print(f"[trainer {worker.rank}] ready", flush=True)
    serve(worker, RPC_PORT_BASE + worker.rank)


if __name__ == "__main__":
    main()
