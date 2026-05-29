#!/bin/zsh
#SBATCH --job-name=etha_benchmark
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=1:00:00

# Surface a torchrun crash instead of exiting 0 + "Benchmark completed".
# Diagnostics below use ${VAR:-} so unset SLURM vars don't trip -u.
set -euo pipefail

# Job information
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Node list: ${SLURM_JOB_NODELIST:-}"
echo "Node ID: ${SLURM_NODEID:-}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID:-}"
echo "============================================"

# NCCL configuration for distributed training
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_TIME_OUT="22"
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NCHANNELS_PER_NET_PEER=4
# export NCCL_DEBUG="INFO"

# Distributed training environment variables
export NODE_RANK=${JOB_COMPLETION_INDEX}
export MASTER_ADDR=${SLURM_JOB_FIRST_NODE_IP}
# Per-allocation port (avoids EADDRINUSE from a stale process on a fixed port).
# cksum is deterministic and MASTER_ADDR is identical across nodes, so all ranks agree.
export MASTER_PORT=$(( 20000 + $(printf '%s' "${MASTER_ADDR}" | cksum | cut -d' ' -f1) % 20000 ))

echo "Node rank: ${NODE_RANK}"
echo "Master address: ${MASTER_ADDR}"
echo "Master port: ${MASTER_PORT}"
echo "Total nodes: ${SLURM_NNODES}"
echo "============================================"
echo "Begin benchmark..."

# Run benchmark with torchrun
pixi run -e dev --frozen --no-install \
    torchrun \
        --nnodes=${SLURM_NNODES} \
        --node-rank=${NODE_RANK} \
        --nproc-per-node=8 \
        --master-addr=${MASTER_ADDR} \
        --master-port=${MASTER_PORT} \
        ./bench/transfer_benchmark.py

echo "============================================"
echo "Benchmark completed"
echo "============================================"
