#!/bin/bash
#SBATCH --job-name=etha-qwen36-dp
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32

# Qwen3.6-35B-A3B(qwen3_5_moe):混合 linear-attn(GatedDeltaNet)+ 每4层 full-attn + MoE + 多模态(visual)。
# trainer AutoModelForCausalLM → Qwen3_5MoeForCausalLM(自动剥 vision/mtp,命名 model.layers.N),
#   manifest 与 trainer named_parameters 天然一致。
# vLLM 是多模态 ForConditionalGeneration(LLM 在 self.language_model 子树),etha 适配(路 A+Y):
#   export 遍历 language_model 子树对齐命名/排除 visual;receive 补回 model.language_model. 前缀;
#   GatedDeltaNet mamba 投影(in_proj_qkv/ba/conv1d/dt_bias/A_log)全副本喂全量,vLLM 自带 mamba
#   loader 按 head 切(tp>1 用)。先 tracer 验命名/shape 对齐 + win 推进,通了再真权重验 Paris。
export TRAIN_NODES=1
export INFER_NODES=1
export ETHA_MODEL=/data/hf/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/7da1103448ba36029c34ce1a9a741dfe93ee0c50
export ETHA_DCP=1
export ETHA_TRACER=${ETHA_TRACER:-0}  # 0=真权重(dcp.load)验 Paris(bucket+缓存+monkey-patch 数值正确性);1=tracer profiling
export ETHA_DP_ATTN=${ETHA_DP_ATTN:-1}
export ETHA_GDN_BACKEND=${ETHA_GDN_BACKEND:-triton}  # GatedDeltaNet prefill 走 triton:vllm env 未装 flashinfer,而 sm_90 默认会硬选 flashinfer 致 EngineCore 崩
# broadcast 子组模式(默认):win-0-stuck 由 trainer reshard 修复(uneven shard 权重 replicate)消除;ETHA_FANOUT=1 可退纯 P2P
export ETHA_GPU_UTIL=${ETHA_GPU_UTIL:-0.6}
export ETHA_PG_TIMEOUT=${ETHA_PG_TIMEOUT:-600}
export ETHA_PROGRESS=${ETHA_PROGRESS:-10}
export ETHA_MAX_LEN=${ETHA_MAX_LEN:-2048}
export ETHA_MAX_BATCHED=${ETHA_MAX_BATCHED:-2048}
export INFER_QUANT=${INFER_QUANT:-}
export ETHA_DECL_DUMP=${ETHA_DECL_DUMP:-/home/jovyan/repos/etha-refactor/etha_decls_qwen36.pkl}  # decl 落持久路径(transfer 前 dump),离线对比两端 build_chunks 的 bcast group 对称性
export ETHA_PROFILE=${ETHA_PROFILE:-/home/jovyan/repos/etha-refactor/etha_trace}  # 真权重+profiler 缓存版:验 Paris + 看 feed 缓存后 load_weights 的 named_parameters/make_expert_mapping 消失
export ETHA_NO_VISION=${ETHA_NO_VISION:-1}  # 跳 vision encoder profile run(--limit-mm-per-prompt image=0),省 init ~60s(transfer/Paris 纯文本不碰 vision)
exec bash scripts/run_e2e_dist.sbatch
