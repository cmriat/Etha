#!/bin/bash
#SBATCH --job-name=etha-gpu-reshard
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -ex
nvidia-smi -L
pixi run -e gpu torchrun --standalone --nproc_per_node=8 scripts/test_gpu.py
