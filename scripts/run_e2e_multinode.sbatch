#!/bin/bash
#SBATCH --job-name=etha-mn
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=64

# 多节点:node 0(JOB_COMPLETION_INDEX=0)跑 trainer(8 rank),node 1 跑 vLLM(8)+ driver。
# etha cross-group host = 头节点 IP(SLURM_JOB_FIRST_NODE_IP),16 ranks 跨 2 节点;
# 端口固定(各节点独占,且 SLURM_JOB_ID 跨节点不一致不能用来派生)。
set -ex
export HF_HOME=/data/hf
export ETHA_MODEL=${ETHA_MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}
export ETHA_HTTP_PORT=52100
export ETHA_VLLM_PORT=52300
export PYTHONPATH=$PWD/examples/megatron_vllm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_SERVER_DEV_MODE=1
export TRITON_CACHE_DIR=/tmp/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
export VLLM_CACHE_ROOT=/tmp/vllm-cache

export TRAINER_WORLD=8
export VLLM_TP=8

if [ "${JOB_COMPLETION_INDEX}" = "0" ]; then
  # ── trainer 节点:8 rank FSDP + HTTP/zmq server(阻塞)──
  pixi run -e fsdp torchrun --standalone --nproc_per_node=8 \
    examples/megatron_vllm/trainer_server.py
else
  # ── inference 节点:vLLM(8,EP)+ driver(本地连 vLLM,经头节点 IP 连 trainer)──
  export ETHA_CROSS_HOST=${SLURM_JOB_FIRST_NODE_IP}
  export ETHA_TRAINER_HOST=${SLURM_JOB_FIRST_NODE_IP}
  pixi run -e vllm vllm serve $ETHA_MODEL \
    --load-format dummy --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
    --enforce-eager --gpu-memory-utilization 0.85 --port $ETHA_VLLM_PORT \
    --worker-extension-cls vllm_side.EthaWorkerExtension \
    --weight-transfer-config '{"backend": "nccl"}' &
  pixi run -e vllm python examples/megatron_vllm/driver.py
fi
