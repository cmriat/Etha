"""671B sharded 加载:meta + FSDP2 shard + dcp.load(HuggingFaceStorageReader)。

DCP 按 checkpoint key 读,但 transformers 5.x 模型是 fused(experts.gate_up_proj
[E,2I,H] / down_proj [E,H,I]),checkpoint 是 per-expert(experts.E.gate_proj 等)→
Missing key。GroupedMoEPlanner 在 set_up_planner 把 fused 本地 shard 拆成 per-expert
key(views into 本地 tensor),DCP 把 per-expert checkpoint 读进 views,fused 张量逐
expert 填满。dense/MLA/norm 名字本就对得上,透传。

expert 全局偏移从 DTensor placements 复合而来(沿所有 Shard(0) 维取 local rank),
不是用扁平 dist.get_rank()——后者只在一维 mesh 下成立,EP/HSDP/dp_replicate 下
会把复制的 rank 误判成不同 expert。一维 fully_shard 时它退化成 rank*n。
"""

import os
import re

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from huggingface_hub import snapshot_download
from torch.distributed.checkpoint import DefaultLoadPlanner, HuggingFaceStorageReader
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard
from transformers import AutoConfig, AutoModelForCausalLM

_GROUPED_RE = re.compile(r"^(.+\.mlp\.experts)\.(gate_up_proj|down_proj)$")


def _explode_grouped_moe(state_dict):
    out = {}
    for key, value in state_dict.items():
        m = _GROUPED_RE.match(key)
        if m is None:
            out[key] = value
            continue
        prefix, role = m.group(1), m.group(2)
        local = value._local_tensor if isinstance(value, DTensor) else value
        e_local = local.shape[0]
        chunk_idx = 0
        if isinstance(value, DTensor):
            mesh = value.device_mesh
            for mesh_dim, pl in enumerate(value.placements):
                if isinstance(pl, Shard) and pl.dim == 0:
                    chunk_idx = chunk_idx * mesh.size(mesh_dim) + mesh.get_local_rank(mesh_dim)
        e_start = chunk_idx * e_local
        if role == "gate_up_proj":
            half = local.shape[1] // 2
            for i in range(e_local):
                out[f"{prefix}.{e_start + i}.gate_proj.weight"] = local[i, :half, :]
                out[f"{prefix}.{e_start + i}.up_proj.weight"] = local[i, half:, :]
        else:
            for i in range(e_local):
                out[f"{prefix}.{e_start + i}.down_proj.weight"] = local[i]
    return out


class GroupedMoEPlanner(DefaultLoadPlanner):
    def set_up_planner(self, state_dict, *args, **kwargs):
        super().set_up_planner(_explode_grouped_moe(state_dict), *args, **kwargs)


def load_sharded(model_name, dtype=torch.bfloat16, tracer=False):
    """meta + FSDP2 shard + DCP load,每 rank 只读自己的 shard(671B 不上单卡)。

    tracer=True 跳过 dcp.load:形状正确、值是 to_empty 垃圾——调形状/带宽用,
    秒起,免去每轮几分钟 NFS 读;值正确性验证时关掉。
    """
    model_dir = model_name if os.path.isdir(model_name) else snapshot_download(model_name, local_files_only=True)
    cfg = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    world = dist.get_world_size()
    rep_mesh = DeviceMesh("cuda", torch.arange(world).view(-1, 1), mesh_dim_names=("replicate", "non_shard"))
    for layer in model.model.layers:
        for sub in layer.modules():  # shape[0]<world(shared_expert_gate [1,H] 等)无法均分 dim0,(world,1) mesh 在 size-1 维分=replicate(susser-tod 式)
            if any(p.dim() and p.shape[0] < world for p in sub.parameters(recurse=False)):
                fully_shard(sub, mesh=rep_mesh)
        fully_shard(layer)
    fully_shard(model)
    model.to_empty(device="cuda")
    if not tracer:
        sd = model.state_dict()
        mm = hasattr(cfg, "text_config")  # 多模态(Qwen3.5/3.6):ckpt LLM 在 model.language_model 下、expert 已 fused(无需 explode)
        if mm:
            sd = {("model.language_model." + k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
        dcp.load(sd, storage_reader=HuggingFaceStorageReader(model_dir),
                 planner=DefaultLoadPlanner() if mm else GroupedMoEPlanner())
    return model
