"""离线 671B 计划 dry-run:纯几何,零 GPU/集群,mac 上即可跑。

meta-init 模型 → manifest;派生两端 placement;对每个权重跑 get_m2m_map;
聚合 routes/bytes、抓 NotImplementedError(不支持的 placement)、查名字 join。
在烧节点前验证 128→32 的计划是否成立。

python scripts/plan_dryrun.py deepseek-ai/DeepSeek-V3 --trainer 128 --tp 8 --ep 4

注:trainer 用 Shard(0) 近似 FSDP(真实是 flatten+pad);divisibility 不整除标为
FSDP-pad 待留意,不当硬错。vLLM placement 用名字规则近似 vllm_side 的模块规则。
"""

import argparse

import torch
from torch.distributed.tensor import Replicate, Shard
from transformers import AutoConfig, AutoModelForCausalLM

from etha import get_m2m_map

R, S = Replicate(), Shard


def vllm_placement(name, dp, tp):
    """名字规则近似 vllm_side 的模块规则(mesh = (1, dp, tp))。"""
    if "experts.gate_up_proj" in name or "experts.down_proj" in name:
        return (R, S(0), S(0))  # MoE:expert 维(EP 借 dp×tp)
    base = name.rsplit(".", 1)[-2] if "." in name else name
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "q_b_proj", "kv_b_proj")):
        return (R, R, S(0))  # column
    if any(k in name for k in ("o_proj", "down_proj")):
        return (R, R, S(1))  # row
    return (R, R, R)  # embed/lm_head/norm/gate/router/kv_a/q_a/bias → replicate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--trainer", type=int, default=128)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--ep", type=int, default=4)
    args = ap.parse_args()
    T, tp, dp = args.trainer, args.tp, args.ep

    cfg = AutoConfig.from_pretrained(args.model)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    tied = getattr(cfg, "tie_word_embeddings", False)

    tmesh = torch.arange(T).reshape(T)  # trainer:1D dp 网格,Shard(0) 近似 FSDP
    vmesh = T + torch.arange(dp * tp).reshape(1, dp, tp)  # vLLM (1, dp, tp)

    from collections import Counter

    n_w = n_route = 0
    total_bytes = 0.0
    errors, pad_flags, repl = [], [], 0
    repl_kinds = Counter()
    repl_bytes = Counter()

    def kind_of(name):
        for k in ("embed_tokens", "lm_head", "kv_a_proj", "q_a_proj", "kv_a_layernorm", "q_a_layernorm",
                  "input_layernorm", "post_attention_layernorm", "q_norm", "k_norm", "mlp.gate.",
                  "e_score_correction_bias", "shared_head"):
            if k in name:
                return k
        if name.endswith(".norm.weight") or name == "model.norm.weight":
            return "model.norm"
        if name.endswith(".bias"):
            return "bias"
        return f"OTHER:{name}"
    for name, p in model.named_parameters():
        if tied and name == "lm_head.weight":
            continue
        n_w += 1
        vpl = vllm_placement(name, dp, tp)
        if all(isinstance(x, Replicate) for x in vpl):
            repl += 1
            k = kind_of(name)
            repl_kinds[k] += 1
            repl_bytes[k] += p.numel() * 2 / 1e9
        try:
            m2m = get_m2m_map(tmesh, (S(0),), vmesh, vpl)
            n_route += len(m2m.routes)
            wbytes = p.numel() * 2  # bf16,全权重一份
            repl_factor = 1
            for i, x in enumerate(vpl):  # 受端按 Replicate 维广播:每个元素落到那些维的所有 rank
                if isinstance(x, Replicate):
                    repl_factor *= int(vmesh.shape[i])
            total_bytes += wbytes * repl_factor
        except Exception as e:
            msg = str(e)
            if "divisi" in msg.lower() or p.shape and p.shape[0] % T:
                pad_flags.append((name, tuple(p.shape), msg[:60]))
            else:
                errors.append((name, tuple(p.shape), str(vpl), f"{type(e).__name__}: {msg[:80]}"))

    print(f"\n=== plan dry-run: {args.model}  trainer={T} → vllm dp={dp} tp={tp} ===")
    print(f"weights: {n_w}  (replicate-fallback: {repl})")
    model_gb = sum(p.numel() for n, p in model.named_parameters() if not (tied and n == "lm_head.weight")) * 2 / 1e9
    print(f"routes : {n_route}")
    print(f"model size (1×): {model_gb:.1f} GB")
    print(f"total received across all infer ranks (含 replicate 广播): {total_bytes / 1e9:.1f} GB  ({total_bytes/1e9/model_gb:.1f}× model)")
    print(f"unsupported placements (HARD): {len(errors)}")
    for e in errors[:15]:
        print("  ", e)
    print(f"non-divisible-by-trainer (FSDP-pad territory): {len(pad_flags)}")
    for f in pad_flags[:10]:
        print("  ", f)
    print("\n=== replicate-fallback 明细(类型: 个数, 总GB/rank) ===")
    for k, c in sorted(repl_kinds.items(), key=lambda x: -repl_bytes[x[0]]):
        print(f"  {k:32s} x{c:<5d} {repl_bytes[k]:.2f} GB")


if __name__ == "__main__":
    main()
