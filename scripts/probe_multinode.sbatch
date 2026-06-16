#!/bin/bash
#SBATCH --job-name=etha-probe
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32

set +e
echo "=== hostname=$(hostname) ip=$(hostname -i) ==="
echo "=== env (rendezvous candidates) ==="
env | grep -iE 'MASTER|FIRST|NODE|INDEX|HEAD|RANK|ADDR|WORLD|JOB|SLURM|POD|HOST' | sort
echo "=== resolve index-0 peer (guess DNS) ==="
getent hosts ${HOSTNAME%-*}-0 2>/dev/null || echo "no -0 dns"
