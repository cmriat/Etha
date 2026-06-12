"""vLLM 收端的 EngineWeightProtocol 实现(每个 worker 进程内,经 worker_extension_cls 注入)。

伪代码:vllm 的 import 与调用标 # pseudo;etha 调用是真实 API。

统一 loader 路线:etha 只解决并行(placement m2m)——收端 buffer 收到的是
「HF 逻辑布局的本 rank shard」;摆放(fuse/permute/数学)交给引擎自己的
load_weights 用真数据跑,纯搬运,与下一组的通信 overlap。

L2(真实执行切分/摆放的 weight_loader)是有限封闭集,与模型数无关:
  Linear 四家            is_sharded_weight=True 跳过 TP narrow(现成,bnb/fairseq2 先例)
  FusedMoE               EP 维天然(per-expert 名喂本地,expert_map 自己消化);
                         EP-off 的 flatten-TP narrow 是缺口
  VocabParallelEmbedding 缺口
  default_weight_loader  无切分,天然 OK
两处缺口 = 对齐 Linear 语义的小上游 PR;落地前这两类权重 fallback 全 Replicate 声明。

quant 模型(process 非平凡):同一个喂法包进 vLLM 自有的 layerwise reload 管线
(逐层物化 → 摆放 → process → copy 回原 kernel storage,CUDA graph 安全),
零自研 staging。
"""

import torch
from torch.distributed.tensor import Shard, Replicate
from vllm.model_executor.utils import set_weight_attrs  # pseudo

# pseudo: vLLM 内部类型与工具
from vllm.model_executor.layers.linear import RowParallelLinear, ColumnParallelLinear  # pseudo
from vllm.model_executor.layers.fused_moe import FusedMoE  # pseudo
from vllm.model_executor.model_loader.reload.layerwise import (  # pseudo
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding  # pseudo

from etha.utils import local_shape  # 声明几何 → 本 rank shard shape(planner 同款几何)

R, S = Replicate(), Shard


class VllmWeightProtocol:
    def __init__(self, model, topo, manifest):
        # topo 由 driver 下发:my_stage_ranks(本 PP stage 的 rank 段)、dp、tp、my_rank。
        # manifest:HF checkpoint index,{hf_name: (shape, dtype)},权威清单。
        #
        # mesh = 持有该权重的 rank 集合:PP 下 vLLM 只构造本 stage 的层,
        # named_modules 天然只枚举到它们,本 stage 的 (R, dp, tp) 格子即 mesh。
        self.model = model
        self.topo = topo
        self.mesh = torch.tensor(topo.my_stage_ranks).reshape(-1, topo.dp, topo.tp)

        # 打标 iff 声明含 Shard:收 shard 的权重 L2 跳 narrow;收 full 的(norm、
        # 缺口 fallback 的 embedding/MoE-EP-off)loader 照常自己 narrow——两者同源,
        # 声明与 loader 行为的一致性内生,逐模型零改动。
        self._shardings = {}
        for module_name, module in model.named_modules():
            rule = self._placement_rule(module)
            for param_name, param in module.named_parameters(recurse=False):
                placements = (rule or {}).get(param_name, (R, R, R))
                if any(isinstance(p, Shard) for p in placements):
                    set_weight_attrs(param, {"is_sharded_weight": True})
                for hf_name in self._hf_names(module_name, param_name):
                    self._shardings[hf_name] = (self.mesh, placements)

        self._buffers = {
            name: torch.empty(
                local_shape(manifest[name][0], self.mesh, self._shardings[name][1], topo.my_rank),
                dtype=manifest[name][1],
                device="cuda",
            )
            for name in manifest
        }  # HF 逻辑布局的本 rank shard;按组分配/复用见 driver 的分组循环

        if self._needs_process():  # quant 等 process 非平凡的配置
            record_metadata_for_reloading(model)

    # ── ② parallel:placement 规则按 layer 类型分发 ────────────────────────
    # 维序 = (replica, dp, tp);replica 维恒 Replicate,dp 维仅对 MoE-EP 是切分维。

    def _placement_rule(self, module):
        if isinstance(module, ColumnParallelLinear):  # qkv/gate_up 也是其子类
            return {"weight": (R, R, S(0)), "bias": (R, R, S(0))}
        if isinstance(module, RowParallelLinear):
            return {"weight": (R, R, S(1)), "bias": (R, R, R)}
        if isinstance(module, VocabParallelEmbedding):
            return {"weight": (R, R, R)}  # 缺口:loader 无 sharded 旁路,
        if isinstance(module, FusedMoE):  # 全 Replicate 收 full,上游 PR 后改 S(0)
            if module.use_ep:  # pseudo: EP 借走 dp×tp 全部格子
                # 嵌套 Shard 假设 expert 连续段分配(linear,默认)。round_robin 是
                # 交错 footprint(_StridedShard 形态,待对拍);EPLB 动态重排 + 冗余
                # 副本,静态声明失效 → 都需 per-expert 粒度声明,此处先 fail loud。
                assert module.expert_placement_strategy == "linear"  # pseudo
                assert not module.enable_eplb  # pseudo
                return {"w13_weight": (R, S(0), S(0)), "w2_weight": (R, S(0), S(0))}
            return {"w13_weight": (R, R, R), "w2_weight": (R, R, R)}  # 缺口同 embedding
        return None  # norm/router 等:全 Replicate

    # ── ① name:融合 param ↔ HF 逻辑权重,两张引擎自带的表,零手抄 ───────────

    def _hf_names(self, module_name, param_name):
        packed = self.model.packed_modules_mapping  # pseudo: {"qkv_proj": ["q_proj",...]}
        stem, leaf = module_name.rsplit(".", 1)
        subs = packed.get(leaf, [leaf])
        vllm_names = [f"{stem}.{sub}.{param_name}" for sub in subs]
        return [self.model.hf_to_vllm_mapper.inverse(n) for n in vllm_names]  # pseudo: 1→1 反向
        # MoE 注:w13/w2 ↔ HF per-expert 名;EP 维不需要跳过——按 per-expert 名
        # 只喂本地 expert(带 global id),loader 的 expert_map 自己消化。

    # ── EngineWeightProtocol ────────────────────────────────────────────────

    def get_sharding(self, hf_name):
        return self._shardings[hf_name]  # 清单上的名字查不到 → KeyError 即 bug

    def local_view(self, hf_name):
        return self._buffers[hf_name]  # 收端落点:HF 布局的本 rank shard

    def process_after_load(self, hf_names):
        """摆放 + (quant 档)process,全部由引擎自己的管线完成。

        直落档(bf16 等,process no-op):裸喂 load_weights,L2 跳 narrow 直落 param。
        quant 档:同一个喂法包进 layerwise 三步——逐层物化 → 摆放 → process →
        copy 回原 kernel storage(地址不变,CUDA graph 安全),vLLM 自有机制。
        """
        shards = ((n, self._buffers[n]) for n in hf_names)
        if self._needs_process():
            initialize_layerwise_reload(self.model)
            self.model.load_weights(shards)
            finalize_layerwise_reload(self.model, self.model.config)  # pseudo
        else:
            self.model.load_weights(shards)

    def _needs_process(self):
        return self.model.quant_config is not None  # pseudo: 配置档判定
