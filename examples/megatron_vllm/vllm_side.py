"""vLLM 收端:worker extension,driver 经 collective_rpc 调,在每个 worker 进程内执行。

统一 loader 路线(设计文档落地节):etha 只管并行——接收 buffer 是 HF 逻辑布局的
本 rank shard;摆放(fuse/GQA 段/数学)由引擎自己的 load_weights 用真数据跑。
打标规则:is_sharded_weight=True iff 声明含 Shard(L2 跳过 TP narrow);
声明全 Replicate 的(norm、embedding 缺口)收 full,loader 照常自己切。
bf16 直落(process no-op);quant 档把同一个喂法包进 layerwise reload,此处未接。
"""

import pickle

import torch
from protocol import build_chunks
from torch.distributed.tensor import Shard, Replicate

from etha import chunk_comm, create_cross_group
from etha.utils import local_shape

R, S = Replicate(), Shard


def _placement_rule(module):
    from vllm.model_executor.layers.linear import RowParallelLinear, ColumnParallelLinear
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    if isinstance(module, ColumnParallelLinear):  # qkv/gate_up 也是其子类
        return {"weight": (R, R, S(0)), "bias": (R, R, S(0))}
    if isinstance(module, RowParallelLinear):
        return {"weight": (R, R, S(1)), "bias": (R, R, R)}
    if isinstance(module, VocabParallelEmbedding):
        return {"weight": (R, R, R)}  # loader 无 sharded 旁路(上游 PR 前收 full)
    return None


class _Api:
    def __init__(self, shardings, buffers):
        self._shardings, self._buffers = shardings, buffers

    def get_sharding(self, name):
        return self._shardings[name]

    def local_view(self, name):
        return self._buffers[name]


class EthaWorkerExtension:
    """混入 vLLM Worker(self.model_runner.model / self.rank 可用)。"""

    def etha_export(self, base_rank, dp):
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.utils import set_weight_attrs

        model = self.model_runner.model
        tp = get_tensor_model_parallel_world_size()
        self._etha_base = base_rank
        mesh = base_rank + torch.arange(dp * tp).reshape(1, dp, tp)
        packed = getattr(model, "packed_modules_mapping", {})
        self._etha_shardings = {}
        for module_name, module in model.named_modules():
            rule = _placement_rule(module)
            for param_name, param in module.named_parameters(recurse=False):
                placements = (rule or {}).get(param_name, (R, R, R))
                if any(isinstance(p, Shard) for p in placements):
                    set_weight_attrs(param, {"is_sharded_weight": True})
                stem, leaf = module_name.rsplit(".", 1)
                for sub in packed.get(leaf, [leaf]):
                    self._etha_shardings[f"{stem}.{sub}.{param_name}"] = (mesh, placements)
        return pickle.dumps(self._etha_shardings)  # 信封:绕过 collective_rpc 的 msgpack 编码

    def etha_init(self, host, port, world, manifest, peer):
        manifest, peer = pickle.loads(manifest), pickle.loads(peer)  # 信封拆封
        rank = self._etha_base + self.rank
        self._etha_buffers = {
            name: torch.empty(local_shape(shape, *self._etha_shardings[name], rank), dtype=dtype, device="cuda")
            for name, (shape, dtype) in manifest.items()
        }
        api = _Api(self._etha_shardings, self._etha_buffers)
        self._etha_group = create_cross_group(host, port, rank, world)
        self._etha_chunks = build_chunks(api, list(manifest), peer, rank, sending=False)

    def etha_transfer(self):
        chunk_comm(self._etha_chunks, group=self._etha_group)
        self.model_runner.model.load_weights(iter(self._etha_buffers.items()))
