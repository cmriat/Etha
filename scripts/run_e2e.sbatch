#!/bin/bash
#SBATCH --job-name=etha-e2e
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -ex
export ETHA_HTTP_PORT=$((40000 + RANDOM % 10000))
export ETHA_VLLM_PORT=$((30000 + RANDOM % 10000))
export ETHA_MODEL=${ETHA_MODEL:-Qwen/Qwen3-0.6B}
export PYTHONPATH=$PWD/examples/megatron_vllm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_SERVER_DEV_MODE=1

export TRITON_CACHE_DIR=/tmp/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
export VLLM_CACHE_ROOT=/tmp/vllm-cache

CUDA_VISIBLE_DEVICES=0,1,2,3 pixi run -e fsdp torchrun --standalone --nproc_per_node=4 \
  examples/megatron_vllm/trainer_server.py &

CUDA_VISIBLE_DEVICES=4,5,6,7 pixi run -e vllm vllm serve $ETHA_MODEL \
  --load-format dummy --tensor-parallel-size 4 --enforce-eager \
  --gpu-memory-utilization 0.6 --port $ETHA_VLLM_PORT \
  --worker-extension-cls vllm_side.EthaWorkerExtension \
  --weight-transfer-config '{"backend": "nccl"}' &

pixi run -e vllm python examples/megatron_vllm/driver.py
