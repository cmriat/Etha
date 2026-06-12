"""Megatron 发端的 EngineWeightProtocol 实现(每个 trainer rank 内)。

伪代码:megatron 的 import 与调用标 # pseudo;etha 调用是真实 API。

Megatron 与 torch-native 的差别:分片是命令式的(写在 layer 代码里),没有
param.placements 可读——②parallel 要一个 converter:从并行配置 + layer 类型
推 placement 声明。这正是引擎矩阵里 Megatron 欠的那格(dev.md M4)。

发端没有 ④ 非仿射(训练不 swizzle);仿射(qkv 去交错)= local_view 返回
变换后的 view,prepare 的 contiguous 物化即"执行变换",etha 核心无感。
"""

import torch
from megatron.core import parallel_state as ps  # pseudo
from torch.distributed.tensor import Shard, Replicate

R, S = Replicate(), Shard


class MegatronWeightProtocol:
    def __init__(self, model, base_rank=0):
        # cross-world rank = base_rank + Megatron global rank(trainer 占头段)。
        #
        # mesh 装配:Megatron rank 序由 initialize_model_parallel 的 order 决定
        # (默认 tp-cp-ep-dp-pp,tp 最内)。PP 不进 placement(per-param mesh,
        # 与 vLLM 侧对称):本 stage 的格子即 mesh——
        #   dense 权重:(dp, tp),dp 维 Replicate(DistributedOptimizer 切的是
        #   optimizer 状态,bf16 param 本体每个 dp rank 完整持有);
        #   MoE 权重:(dp_moe, ep, etp)——EP 借 dp 的格子(dp = dp_moe × ep),
        #   ETP 可同时开(Megatron 有 EP×ETP,placement 三维嵌套,etha 已覆盖)。
        self.model = model
        pp, dp, tp = (
            ps.get_pipeline_model_parallel_world_size(),
            ps.get_data_parallel_world_size(),
            ps.get_tensor_model_parallel_world_size(),
        )  # pseudo
        ranks = base_rank + torch.arange(pp * dp * tp).reshape(pp, dp, tp)
        self.mesh = ranks[ps.get_pipeline_model_parallel_rank()]  # pseudo: 本 stage (dp, tp)
        ep = ps.get_expert_model_parallel_world_size()  # pseudo
        etp = ps.get_expert_tensor_parallel_world_size()  # pseudo
        self.moe_mesh = self.mesh.reshape(dp // ep, ep, etp)  # EP 借 dp 格子

        self._shardings = {}
        self._views = {}
        for module_name, module in model.named_modules():
            rule = self._placement_rule(module)
            for param_name, param in module.named_parameters(recurse=False):
                mesh, placements = (rule or {}).get(param_name, (self.mesh, (R, R)))
                for hf_name, view in self._decompose(module_name, module, param_name, param):
                    self._shardings[hf_name] = (mesh, placements)
                    self._views[hf_name] = view

    # ── ② parallel converter:Megatron layer 类型 → placement 声明 ──────────
    # 这张表就是 Megatron 命令式分片的声明化,M4 的核心;白名单 fail loud。

    def _placement_rule(self, module):
        from megatron.core.tensor_parallel import (  # pseudo
            RowParallelLinear,
            ColumnParallelLinear,
            VocabParallelEmbedding,
        )
        from megatron.core.transformer.moe.experts import GroupedMLP  # pseudo

        if isinstance(module, ColumnParallelLinear):  # linear_qkv / linear_fc1
            return {"weight": (self.mesh, (R, S(0))), "bias": (self.mesh, (R, S(0)))}
        if isinstance(module, RowParallelLinear):  # linear_proj / linear_fc2
            return {"weight": (self.mesh, (R, S(1))), "bias": (self.mesh, (R, R))}
        if isinstance(module, VocabParallelEmbedding):
            return {"weight": (self.mesh, (R, S(0)))}
        if isinstance(module, GroupedMLP):  # MoE experts(grouped)
            return {
                "weight1": (self.moe_mesh, (R, S(0), S(2))),  # (E, H, I):EP 切 expert,ETP 切 I
                "weight2": (self.moe_mesh, (R, S(0), S(1))),  # (E, I, H):ETP 切 I
            }
        return None  # norm 等:全 Replicate

    # ── ① name + ③ 仿射:Megatron param → (HF 名, 发送 view) ────────────────
    #
    # 名字是 trainer→HF 的固有成本(verl 同样手写);view 是发送侧仿射的全部——
    # 返回"变换后的 view",prepare 的 contiguous 物化即执行变换,etha 核心无感。

    _NAME = {
        "linear_qkv": ("self_attn", ["q_proj", "k_proj", "v_proj"]),
        "linear_proj": ("self_attn", ["o_proj"]),
        "linear_fc1": ("mlp", ["gate_proj", "up_proj"]),
        "linear_fc2": ("mlp", ["down_proj"]),
    }

    def _global_name(self, module_name):
        """Megatron 层号是 stage 局部的,HF 名用全局编号——PP 在名字上的唯一手工活。
        VPP(interleaved)时 model 是 chunk list,offset per chunk 算,此处略。"""
        from megatron.core.transformer.utils import get_transformer_layer_offset  # pseudo

        local_idx = int(module_name.split("layers.")[1].split(".")[0])
        return module_name.replace(
            f"layers.{local_idx}", f"layers.{local_idx + get_transformer_layer_offset(self.model.config)}"
        ).replace("decoder", "model")

    def _decompose(self, module_name, module, param_name, param):
        prefix = self._global_name(module_name).rsplit(".", 1)[0]
        leaf = module_name.rsplit(".", 1)[1]

        if leaf == "linear_qkv":
            # mcore 交错布局:dim0 = [q × qpg, k, v] × ng(按 GQA group 交错);
            # TP 切 group,本 rank 持 ng_local 组,shard 内同构交错。
            # 去交错 = 纯 strided view,prepare 物化 —— 发送侧仿射的实证。
            cfg = module.config  # pseudo
            ng = cfg.num_query_groups // ps.get_tensor_model_parallel_world_size()  # pseudo
            qpg = cfg.num_attention_heads // cfg.num_query_groups
            hd = cfg.kv_channels
            w = param.view(ng, qpg + 2, hd, -1)
            for sub, seg in zip(["q_proj", "k_proj", "v_proj"], [w[:, :qpg], w[:, qpg], w[:, qpg + 1]], strict=True):
                yield f"{prefix}.self_attn.{sub}.{param_name}", seg.reshape(-1, param.shape[-1])
        elif leaf == "linear_fc1":  # [gate; up] 两段连续拼接
            half = param.shape[0] // 2
            yield f"{prefix}.mlp.gate_proj.{param_name}", param[:half]
            yield f"{prefix}.mlp.up_proj.{param_name}", param[half:]
        elif leaf in self._NAME:
            sub_prefix, subs = self._NAME[leaf]
            yield f"{prefix}.{sub_prefix}.{subs[0]}.{param_name}", param.data
        elif param_name in ("weight1", "weight2"):  # GroupedMLP:清单约定 grouped 名
            hf = {"weight1": "gate_up_proj", "weight2": "down_proj"}[param_name]
            yield f"{prefix}.experts.{hf}", param.data.transpose(-2, -1)  # mcore (…,H,I) → HF (…,I,H)
        else:
            yield f"{self._global_name(module_name)}.{param_name}", param.data

    # ── EngineWeightProtocol ────────────────────────────────────────────────

    def get_sharding(self, hf_name):
        return self._shardings[hf_name]  # 清单上的名字查不到 → KeyError 即 bug

    def local_view(self, hf_name):
        return self._views[hf_name]  # 发送 shard(变换已编码在 view 里)

    def process_after_load(self, hf_names):
        pass  # 发端无 ④ 非仿射(训练不 swizzle)
