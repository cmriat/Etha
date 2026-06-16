#!/bin/bash
#SBATCH --job-name=etha-probe
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -x
echo "NODEID=${SLURM_NODEID} NPROCS=${SLURM_NPROCS} NODELIST=${SLURM_NODELIST} hostname=$(hostname) ip=$(hostname -i)"
echo "FIRST_NODE_IP=${FIRST_NODE_IP:-unset}  MASTER_ADDR=${MASTER_ADDR:-unset}"
nvidia-smi -L
