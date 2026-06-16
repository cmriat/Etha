#!/bin/bash
#SBATCH --job-name=etha-probe
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32

set +e
echo "=== idx=${JOB_COMPLETION_INDEX} host=$(hostname) ip=$(hostname -i) ==="
echo "NODELIST=${SLURM_JOB_NODELIST}"
# 解析 NODELIST 每个 DNS 名,看跨节点能不能寻址(Ray/torchrun 多节点需要)
echo "${SLURM_JOB_NODELIST}" | tr ',' '\n' | while read n; do
  echo "resolve [$n] -> $(getent hosts "$n" | awk '{print $1}' || echo FAIL)"
done
