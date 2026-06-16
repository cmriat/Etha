"""FSDP2 trainer:torchrun 起 N 卡,加载真模型 fully_shard,提供 collective RPC。

driver 经 rpc.CollectiveClient 调下面四个方法,与 vLLM 侧的 worker extension
对称(README 时序)。
"""

import base64
import os
import pickle

import torch
import torch.distributed as dist
from fsdp_side import FsdpWeightProtocol
from protocol import EthaTrainerEngine
from rpc import serve
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM


class TrainerWorker:
    def __init__(self):
        model = AutoModelForCausalLM.from_pretrained(
            os.environ.get("ETHA_MODEL", "Qwen/Qwen3-0.6B"), dtype=torch.bfloat16
        ).cuda()
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)
        self.api = FsdpWeightProtocol(model)
        self.rank = dist.get_rank()

    def etha_export(self, base_rank):
        self.api.base_rank = base_rank          # cross-world 记账是 driver 的决策,显式注入
        decl = {n: self.api.get_sharding(n) for n in self.api._params}
        return base64.b64encode(pickle.dumps(decl)).decode()

    def init_weight_transfer_engine(self, init_info):
        self.engine = EthaTrainerEngine(self.api, self.rank)
        self.engine.init_transfer_engine(self.engine.init_info_cls(**init_info))

    def update_weights(self, update_info):
        self.engine.update_weights(update_info)


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    worker = TrainerWorker()
    print(f"[trainer {worker.rank}] ready", flush=True)
    serve(worker, worker.rank, dist.get_world_size())


if __name__ == "__main__":
    main()
