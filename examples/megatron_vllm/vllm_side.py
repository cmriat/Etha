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
import math
import pickle
from dataclasses import dataclass

import torch
from protocol import EthaInitInfo, build_chunks
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
        # layerwise reload 按 record 快照重建 param(__dict__ 拷回)——启动时的旧快照
        # 会盖掉上面打的标,重新 record 让快照带上它们。
        from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

        record_metadata_for_reloading(model)
        return base64.b64encode(pickle.dumps(shardings)).decode()


# ── 传输引擎(零 model 依赖,官方插件位)────────────────────────────────────

from vllm.distributed.weight_transfer.base import (  # noqa: E402
    WeightTransferEngine,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory  # noqa: E402


@dataclass
class EthaUpdateInfo(WeightTransferUpdateInfo):
    pass  # plan 缓存于 init,每轮无新元数据;is_checkpoint_format 继承基类


class _Api:
    def __init__(self, shardings):
        self._shardings = shardings

    def get_sharding(self, name):
        return self._shardings[name]


class EthaWeightTransferEngine(WeightTransferEngine[EthaInitInfo, EthaUpdateInfo]):
    init_info_cls = EthaInitInfo
    update_info_cls = EthaUpdateInfo

    def init_transfer_engine(self, init_info):
        self._manifest = dec(init_info.manifest)
        self._shardings = dec(init_info.self_decl)
        self._peer = dec(init_info.peer_decl)
        self._rank = init_info.base_rank + self.parallel_config.rank
        self._group = create_cross_group(init_info.host, init_info.port, self._rank, init_info.world)

    def receive_weights(self, update_info, load_weights):
        # 全流式:dst buffer 在执行流中按需分配(target_alloc),某权重的 chunks
        # 收齐即喂 loader 并释放(on_complete)——峰值 = 在飞窗口,与模型大小无关。
        targets = {
            name: (local_shape(shape, *self._shardings[name], self._rank), dtype)
            for name, (shape, dtype) in self._manifest.items()
        }
        chunks = build_chunks(
            _Api(self._shardings), list(self._manifest), self._peer, self._rank, sending=False, targets=targets
        )
        full = sum(math.prod(shape) * dtype.itemsize for shape, dtype in targets.values()) / 1e9

        # 隔离接收 buffer 高水位(流式真正控制的量)——总显存被模型 re-materialize 污染,
        # 不能用。计数器在 alloc 加、on_complete 减,峰值即同时在飞的接收 buffer。
        live = [0.0]
        peak = [0.0]

        def alloc(n):
            buf = torch.empty(targets[n][0], dtype=targets[n][1], device="cuda")
            live[0] += buf.nbytes / 1e9
            peak[0] = max(peak[0], live[0])
            return buf

        def complete(n, buf):
            load_weights([(n, buf)])
            live[0] -= buf.nbytes / 1e9

        chunk_comm(chunks, group=self._group, target_alloc=alloc, on_complete=complete)
        if self.parallel_config.rank == 0:
            print(f"[etha recv] in-flight buffer peak {peak[0]:.3f} GB vs full-shard {full:.3f} GB", flush=True)

    def shutdown(self):
        pass

    @staticmethod
    def trainer_send_weights(iterator, trainer_args):
        raise NotImplementedError("trainer side runs its own server, see trainer_server.py")


# WeightTransferConfig.backend 是封闭 Literal["nccl","ipc"](pydantic 校验拒绝
# 第三方名)——与 factory 的注册制矛盾,上游 PR 应放开为注册名。example 过渡:
# 配置报 "nccl" 过校验,registry 槽位覆写成 etha(extension import 早于 create_engine)。
WeightTransferEngineFactory._registry["nccl"] = lambda: EthaWeightTransferEngine
