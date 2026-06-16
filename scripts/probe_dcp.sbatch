#!/bin/bash
#SBATCH --job-name=etha-dcp-probe
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -ex
export HF_HOME=/data/hf
pixi run -e fsdp torchrun --standalone --nproc_per_node=2 \
  scripts/probe_dcp_load.py "${ETHA_MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
