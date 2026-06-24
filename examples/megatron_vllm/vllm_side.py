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

from etha import chunk_comm, create_broadcast_subgroups, create_cross_group
from etha.utils import local_shape

R, S = Replicate(), Shard


def dec(s):
    return pickle.loads(base64.b64decode(s))


def _to_hf(name):  # CausalLM(manifest)命名 → vLLM 多模态 load_weights 期望的 HF 命名:LLM 在 model.language_model 下
    return "model.language_model." + name[6:] if name.startswith("model.") else name


# ── 元数据接口(需要 model,经 worker_extension 暴露)───────────────────────


def _placement_rule(module, module_name):
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    if ".linear_attn" in module_name and not module_name.endswith("out_proj"):
        return None  # GatedDeltaNet conv1d/in_proj_*/dt_bias/A_log 切法非标准(分段/per-head):全副本喂全量,留 vLLM 自带 mamba weight_loader 按 head 切(tp>1)
    if getattr(module, "disable_tp", False):  # MLA fused_qkv_a_proj:Column 子类但 disable_tp,全副本不切
        return {"weight": (R, R, R), "bias": (R, R, R)}
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
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.utils import set_weight_attrs

        root = self.model_runner.model
        model = getattr(root, "language_model", root)  # 多模态:LLM 在 language_model 子树,遍历它令命名对齐 manifest(model.layers.N)、排除 visual
        if model is not root:
            self.weight_transfer_engine._lm_subtree = True  # receive 喂权重时补回 HF 的 model.language_model. 前缀
        tp = get_tensor_model_parallel_world_size()
        dp = self.vllm_config.parallel_config.data_parallel_size
        mesh = base_rank + torch.arange(dp * tp).reshape(1, dp, tp)
        packed = getattr(model, "packed_modules_mapping", {})
        shardings = {}
        for module_name, module in model.named_modules():
            if isinstance(module, FusedMoE):
                shardings.update(self._moe_shardings(module_name, module, mesh))
                continue
            rule = _placement_rule(module, module_name)
            for param_name, param in module.named_parameters(recurse=False):
                placements = (rule or {}).get(param_name, (R, R, R))
                if tp > 1 and any(isinstance(p, Shard) for p in placements):
                    # tp>1 喂本 rank shard 需 is_sharded_weight 旁路 narrow;tp=1 喂全量、用 vLLM 原
                    # weight_loader(narrow 全量=全量 + copy_)——否则 is_sharded_weight 的 direct-assign
                    # 不走 copy_,layerwise reload 的 CopyCounter 数 0→load_numel=0→clobber 真权重致乱码。
                    set_weight_attrs(param, {"is_sharded_weight": True})
                    param.weight_loader = module.weight_loader
                stem, _, leaf = module_name.rpartition(".")  # 顶层模块(lm_head)无 ".",stem=""
                for sub in packed.get(leaf, [leaf]):
                    full = f"{stem}.{sub}.{param_name}" if stem else f"{sub}.{param_name}"
                    shardings[full] = (mesh, placements, param.dtype)
        # tp>1 的 is_sharded_weight 标需重 record 进快照让 restore 拷回。但 re-record 发生在启动
        # process 之后,会把 kernel-format(process 后形态)写进 restore_metadata,污染 layerwise 退 meta
        # 的基线(materialize 成 kernel-format → load raw 错位)。_initialize_model 在 model 构造、
        # pre-process 时已 record 过 model-format 快照,tp=1 直接用它,绝不 re-record。
        if tp > 1:
            from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

            record_metadata_for_reloading(root)
        return base64.b64encode(pickle.dumps(shardings)).decode()

    def _moe_shardings(self, module_name, module, mesh):
        """融合 expert 权重:expert 维(dim0)reshard,名字用 transformers 融合名。

        transformers 5.x 和 vLLM 的 expert 张量布局完全相同(experts.gate_up_proj
        ↔ w13_weight,内层一致),所以就是 dim0 reshard。EP 借 dp×tp 格子切 expert 维
        → (R, S(0), S(0))。名字 experts.gate_up_proj/down_proj 与 manifest(transformers
        reference)一致;喂时 receive_weights 再切 per-expert 给 native loader。
        """
        assert module.expert_placement_strategy == "linear" and not module.enable_eplb
        return {
            f"{module_name}.gate_up_proj": (mesh, (R, S(0), S(0)), module.w13_weight.dtype),
            f"{module_name}.down_proj": (mesh, (R, S(0), S(0)), module.w2_weight.dtype),
        }

    def etha_update_native(self, update_info):
        # vLLM layerwise reload 在 leaf param.weight_loader 挂 online wrap 延迟 process,但多模态嵌套
        # (ForConditionalGeneration→language_model→model→leaf)每层 child 各有 load_weights,AutoWeightsLoader
        # 逐层调 child.load_weights、绕过 leaf 的 wrap,使 online loader 计数永不触发、process 永不跑致乱码;
        # 退 meta 又让 receive 没覆盖到的 tensor 残留 meta。改为复刻 base_loader 但全程不碰 meta:原地把 lm
        # 子树 param.data 重置为 pre-process 的 model-format empty(清启动 process 的 kernel 排布,保 param 对象
        # 与 weight_loader)→ etha copy_ 真值 → process 一次 shuffle 回 kernel-format。buffer/visual 不碰。
        from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        eng = self.weight_transfer_engine
        root = self.model_runner.model
        lm = getattr(root, "language_model", root)
        with torch.device(self.device):
            for layer in lm.modules():
                rparams, _ = get_layerwise_info(layer).restore_metadata
                cparams = dict(layer.named_parameters(recurse=False))
                for name, meta_p in rparams.items():
                    if name in cparams:
                        cparams[name].data = torch.empty(tuple(meta_p.shape), dtype=meta_p.dtype, device=self.device)
            emaps = [(m, m._expert_map) for m in lm.modules() if getattr(m, "_expert_map", None) is not None]
            for m, em in emaps:  # _expert_map 搬 CPU:weight_loader 的 .item() 不再 GPU→CPU sync 卡 chunk_comm 主循环
                m._expert_map = em.cpu()
            # vLLM Qwen3_5Model.load_weights 每 call 都 dict(named_parameters())(~5ms)+ get_expert_mapping()(make_expert_params_mapping ~5ms),
            # 它假设 startup 一次喂全部、build 一次摊销;etha per-weight 449 call → 重建 449 次 ~4.5s 纯浪费 → patch 成缓存,receive 后还原
            mdl = getattr(lm, "model", lm)
            cached_params, cached_emap = list(mdl.named_parameters()), mdl.get_expert_mapping()
            mdl.named_parameters = lambda *a, **k: iter(cached_params)
            mdl.get_expert_mapping = lambda: cached_emap
            eng.receive_weights(eng.parse_update_info(update_info), root.load_weights)
            del mdl.named_parameters, mdl.get_expert_mapping
            for m, em in emaps:
                m._expert_map = em
            process_weights_after_loading(lm, self.vllm_config.model_config, self.device)


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
    _lm_subtree = False  # 多模态:etha_export 探测到 language_model 子树时置位

    def init_transfer_engine(self, init_info):
        self._manifest = dec(init_info.manifest)
        self._shardings = dec(init_info.self_decl)
        self._peer = dec(init_info.peer_decl)
        pc = self.parallel_config
        self._ep_rank = pc.data_parallel_rank * pc.world_size + pc.rank  # EP 在 dp×tp 格子里的 mesh-local 位(含 dp 偏移);DP 下 parallel_config.rank=0 不含 dp,expert global id 必须用它
        self._rank = init_info.base_rank + self._ep_rank
        self._group, store = create_cross_group(init_info.host, init_info.port, self._rank, init_info.world)
        self._chunks, bcast = build_chunks(_Api(self._shardings), self._manifest, self._peer, self._rank, sending=False)
        self._subgroups = create_broadcast_subgroups(store, self._rank, bcast)
        self._targets = {
            name: (local_shape(shape, *self._shardings[name][:2], self._rank), self._peer[name][2])
            for name, shape in self._manifest.items()
        }

    def receive_weights(self, update_info, load_weights):
        # 全流式:dst buffer 在执行流中按需分配(target_alloc),某权重的 chunks
        # 收齐即喂 loader 并释放(on_complete)——峰值 = 在飞窗口,与模型大小无关。
        # buffer 几何从 manifest 全局 shape + 本端 placement;dtype 从源(peer)声明。
        targets = self._targets
        chunks = self._chunks

        live = [0.0]
        peak = [0.0]

        def alloc(n):
            buf = torch.empty(targets[n][0], dtype=targets[n][1], device="cuda")
            live[0] += buf.nbytes / 1e9
            peak[0] = max(peak[0], live[0])
            return buf

        def complete(n, buf):
            self._feed(n, buf, load_weights)
            live[0] -= buf.nbytes / 1e9

        chunk_comm(chunks, group=self._group, subgroups=self._subgroups, target_alloc=alloc, on_complete=complete)
        if self.parallel_config.rank == 0:
            full = sum(math.prod(s) * d.itemsize for s, d in targets.values()) / 1e9
            print(f"[etha recv] in-flight buffer peak {peak[0]:.3f} GB vs full-shard {full:.3f} GB", flush=True)

    def _feed(self, name, buf, load_weights):
        """MoE 融合 buffer 喂时切 per-expert(贴合 native 的 per-expert loader);
        dense 直接喂。buf 是本 rank 的本地 expert 融合块,全局 expert id 由
        ep_rank(EP 借 dp×tp 格子 → == parallel_config.rank)× local 推出。"""
        if self._lm_subtree:
            load_weights = lambda it, f=load_weights: f([(_to_hf(n), t) for n, t in it])
        self._feed_impl(name, buf, load_weights)

    def _feed_impl(self, name, buf, load_weights):
        if name.endswith(".experts.gate_up_proj"):
            stem = name[: -len(".gate_up_proj")]
            inter, local = buf.shape[1] // 2, buf.shape[0]
            items = []
            for i in range(local):
                g = self._ep_rank * local + i
                items.append((f"{stem}.{g}.gate_proj.weight", buf[i, :inter]))
                items.append((f"{stem}.{g}.up_proj.weight", buf[i, inter:]))
            load_weights(items)  # 一次喂全部 local expert:load_weights→_load_module 递归遍历 model 仅 1 次(原 for 每 expert 一次=256× 遍历)
        elif name.endswith(".experts.down_proj"):
            stem = name[: -len(".down_proj")]
            local = buf.shape[0]
            load_weights([(f"{stem}.{self._ep_rank * local + i}.down_proj.weight", buf[i]) for i in range(local)])
        else:
            load_weights([(name, buf)])

    def shutdown(self):
        pass

    @staticmethod
    def trainer_send_weights(iterator, trainer_args):
        raise NotImplementedError("trainer side runs its own server, see trainer_server.py")


# WeightTransferConfig.backend 是封闭 Literal["nccl","ipc"](pydantic 校验拒绝
# 第三方名)——与 factory 的注册制矛盾,上游 PR 应放开为注册名。example 过渡:
# 配置报 "nccl" 过校验,registry 槽位覆写成 etha(extension import 早于 create_engine)。
WeightTransferEngineFactory._registry["nccl"] = lambda: EthaWeightTransferEngine
