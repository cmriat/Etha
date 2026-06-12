#!/bin/bash
#SBATCH --job-name=etha-e2e
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -ex
export ETHA_MODEL=${ETHA_MODEL:-Qwen/Qwen3-0.6B}
export PYTHONPATH=$PWD/examples/megatron_vllm
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export TRITON_CACHE_DIR=/tmp/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
export VLLM_CACHE_ROOT=/tmp/vllm-cache

CUDA_VISIBLE_DEVICES=0,1,2,3 pixi run -e fsdp torchrun --standalone --nproc_per_node=4 \
  examples/megatron_vllm/trainer_server.py &
trainer_pid=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 pixi run -e vllm python examples/megatron_vllm/driver.py
