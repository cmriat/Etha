"""vLLM 收端:官方 WeightTransferEngine 插件 + 薄 worker extension。

职责分界:
  EthaWeightTransferEngine —— 纯传输(cross group、buffers、chunk_comm),零 model
    依赖;init 走 worker 官方入口 init_weight_transfer_engine,每轮走 update_weights
    (is_checkpoint_format=True 时 worker 自动包 layerwise reload,quant 管线免费)。
  EthaWorkerExtension.etha_export —— 引擎的元数据接口(打标 + placement 声明),
    需要 model,与传输无关;上游化即 get_sharding。
engine 所需的自声明由 driver 从 export 结果回灌进 init_info,字段全显式
(EthaInitInfo 即 init_info 的 schema)。
"""

import base64
import pickle
from dataclasses import dataclass

import torch
from protocol import build_chunks
from torch.distributed.tensor import Replicate, Shard

from etha import chunk_comm, create_cross_group
from etha.utils import local_shape

R, S = Replicate(), Shard


def dec(s):
    return pickle.loads(base64.b64decode(s))


# ── 元数据接口(需要 model,经 worker_extension 暴露)───────────────────────


def _placement_rule(module):
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    if isinstance(module, ColumnParallelLinear):  # qkv/gate_up 也是其子类
        return {"weight": (R, R, S(0)), "bias": (R, R, S(0))}
    if isinstance(module, RowParallelLinear):
        return {"weight": (R, R, S(1)), "bias": (R, R, R)}
    if isinstance(module, VocabParallelEmbedding):
        return {"weight": (R, R, R)}  # loader 无 sharded 旁路(上游 PR 前收 full)
    return None


class EthaWorkerExtension:
    def etha_export(self, base_rank):
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.utils import set_weight_attrs

        model = self.model_runner.model
        tp = get_tensor_model_parallel_world_size()
        dp = self.vllm_config.parallel_config.data_parallel_size
        mesh = base_rank + torch.arange(dp * tp).reshape(1, dp, tp)
        packed = getattr(model, "packed_modules_mapping", {})
        shardings = {}
        for module_name, module in model.named_modules():
            rule = _placement_rule(module)
            for param_name, param in module.named_parameters(recurse=False):
                placements = (rule or {}).get(param_name, (R, R, R))
                if any(isinstance(p, Shard) for p in placements):
                    # is_sharded_weight 只有 v1 weight_loader 检查;bf16 走 v2
                    # (parameter.py 的 load_* 无此旁路)——绑回 v1,上游 PR 应补 v2。
                    set_weight_attrs(param, {"is_sharded_weight": True})
                    param.weight_loader = module.weight_loader
                stem, leaf = module_name.rsplit(".", 1)
                for sub in packed.get(leaf, [leaf]):
                    shardings[f"{stem}.{sub}.{param_name}"] = (mesh, placements)
        return base64.b64encode(pickle.dumps(shardings)).decode()


# ── 传输引擎(零 model 依赖,官方插件位)────────────────────────────────────

from vllm.distributed.weight_transfer.base import (  # noqa: E402
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory  # noqa: E402


@dataclass
class EthaInitInfo(WeightTransferInitInfo):
    host: str
    port: int
    world: int
    base_rank: int
    manifest: str  # base64(pickle):{hf_name: (shape, dtype)},HF index 权威清单
    self_decl: str  # base64(pickle):本端声明(etha_export 产出,driver 回灌)
    peer_decl: str  # base64(pickle):对端声明


@dataclass
class EthaUpdateInfo(WeightTransferUpdateInfo):
    pass  # plan 缓存于 init,每轮无新元数据;is_checkpoint_format 继承基类


class _Api:
    def __init__(self, shardings, buffers):
        self._shardings, self._buffers = shardings, buffers

    def get_sharding(self, name):
        return self._shardings[name]

    def local_view(self, name):
        return self._buffers[name]


class EthaWeightTransferEngine(WeightTransferEngine[EthaInitInfo, EthaUpdateInfo]):
    init_info_cls = EthaInitInfo
    update_info_cls = EthaUpdateInfo

    def init_transfer_engine(self, init_info):
        manifest, shardings, peer = dec(init_info.manifest), dec(init_info.self_decl), dec(init_info.peer_decl)
        rank = init_info.base_rank + self.parallel_config.rank
        self._buffers = {
            name: torch.empty(local_shape(shape, *shardings[name], rank), dtype=dtype, device="cuda")
            for name, (shape, dtype) in manifest.items()
        }
        self._group = create_cross_group(init_info.host, init_info.port, rank, init_info.world)
        self._chunks = build_chunks(_Api(shardings, self._buffers), list(manifest), peer, rank, sending=False)

    def receive_weights(self, update_info, load_weights):
        chunk_comm(self._chunks, group=self._group)
        load_weights(list(self._buffers.items()))

    def shutdown(self):
        pass

    @staticmethod
    def trainer_send_weights(iterator, trainer_args):
        raise NotImplementedError("trainer side runs its own server, see trainer_server.py")


WeightTransferEngineFactory.register_engine("etha", EthaWeightTransferEngine)
