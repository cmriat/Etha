"""探针:验 meta + FSDP2 shard + dcp.load(HuggingFaceStorageReader) 能否加载,
尤其 MoE 的 per-expert(safetensors)↔ fused(transformers 5.x)命名是否需要 planner。

torchrun --standalone --nproc_per_node=2 scripts/probe_dcp_load.py <model>
"""

import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from huggingface_hub import snapshot_download
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.fsdp import fully_shard
from transformers import AutoConfig, AutoModelForCausalLM

model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-ai/DeepSeek-V2-Lite-Chat"
dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

model_dir = snapshot_download(model, local_files_only=True)
if rank == 0:
    print(f"model_dir={model_dir}", flush=True)

cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
with torch.device("meta"):
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, trust_remote_code=True)
for layer in m.model.layers:
    fully_shard(layer)
fully_shard(m)
m.to_empty(device="cuda")

sd = m.state_dict()
if rank == 0:
    moe = [k for k in sd if "experts" in k][:4]
    print(f"state_dict has {len(sd)} keys; MoE-ish keys: {moe}", flush=True)

try:
    dcp.load(sd, storage_reader=HuggingFaceStorageReader(model_dir))
    # 抽查:几个权重非零(加载成功),MoE 融合权重也查
    bad = [k for k, v in sd.items() if v.to_local().abs().sum().item() == 0] if hasattr(next(iter(sd.values())), "to_local") else []
    if rank == 0:
        sample = next(k for k in sd if "experts.gate_up_proj" in k or "mlp" in k)
        v = sd[sample]
        loc = v.to_local() if hasattr(v, "to_local") else v
        print(f"DCP LOAD OK; sample {sample} nonzero={loc.abs().sum().item() > 0}; all-zero keys: {len(bad)}", flush=True)
except Exception as e:
    if rank == 0:
        print(f"DCP LOAD FAILED: {type(e).__name__}: {e}", flush=True)

dist.barrier()
dist.destroy_process_group()
