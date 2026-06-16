#!/bin/bash
#SBATCH --job-name=etha-dist
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=64

# 多节点双边:前 TRAIN_NODES 个节点跑 trainer(多节点 torchrun FSDP),
# 其余 INFER_NODES 个跑 vLLM(TP 跨节点,--nnodes/--node-rank/--master-addr,mp 无 ray)。
# 节点 IP 用 kubectl 发现(NODELIST DNS 不稳)。infer head 跑 API server + driver。
set -ex
export HF_HOME=/data/hf
export TRAIN_NODES=${TRAIN_NODES:-2}
export INFER_NODES=${INFER_NODES:-2}
export ETHA_MODEL=${ETHA_MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}
export ETHA_HTTP_PORT=52100
export ETHA_VLLM_PORT=52300
export PYTHONPATH=$PWD/examples/megatron_vllm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_SERVER_DEV_MODE=1
export TRITON_CACHE_DIR=/tmp/triton-cache TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache VLLM_CACHE_ROOT=/tmp/vllm-cache

# ── 节点发现(kubectl pod IP,按 pod 名排序)──
GROUP_PREFIX=$(echo "$SLURM_JOB_NODELIST" | sed 's/-[0-9]\..*//')
for i in {1..60}; do
  ips=($(kubectl get pods -o wide 2>/dev/null | grep "$GROUP_PREFIX" | sort | awk 'NF>=6 && $6!~/<none>/ && $6~/[0-9]/ {print $6}'))
  [ ${#ips[@]} -eq $SLURM_NNODES ] && break
  sleep 2
done
[ ${#ips[@]} -ne $SLURM_NNODES ] && { echo "node discovery failed: ${#ips[@]}/$SLURM_NNODES"; exit 1; }
export NODE_IPS=$(IFS=,; echo "${ips[*]}")
RANK=${JOB_COMPLETION_INDEX}
TRAIN_HEAD=${ips[0]}
INFER_HEAD=${ips[$TRAIN_NODES]}
echo "NODE_IPS=$NODE_IPS RANK=$RANK TRAIN_HEAD=$TRAIN_HEAD INFER_HEAD=$INFER_HEAD"

IFACE=$(ip -o -4 route show to default | awk '{print $5}' | head -1)
export NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE

if [ "$RANK" -lt "$TRAIN_NODES" ]; then
  # ── trainer 节点:多节点 torchrun FSDP ──
  pixi run -e fsdp torchrun --nnodes $TRAIN_NODES --node-rank $RANK \
    --master-addr $TRAIN_HEAD --master-port 52000 --nproc_per_node 8 \
    examples/megatron_vllm/trainer_server.py
else
  # ── inference 节点:vLLM TP 跨节点 ──
  IRANK=$(( RANK - TRAIN_NODES ))
  TP=$(( INFER_NODES * 8 ))
  if [ "$IRANK" -ne 0 ]; then
    # 非 head:headless worker 前台跑(无 wait,kjobctl 不支持 wait)
    exec pixi run -e vllm vllm serve $ETHA_MODEL \
      --load-format dummy --trust-remote-code --enable-expert-parallel --enforce-eager \
      --tensor-parallel-size $TP --nnodes $INFER_NODES --node-rank $IRANK --master-addr $INFER_HEAD \
      --gpu-memory-utilization 0.85 --headless \
      --worker-extension-cls vllm_side.EthaWorkerExtension \
      --weight-transfer-config '{"backend": "nccl"}'
  fi
  # infer head:serve 后台 + driver(本地连 vLLM,经 train head 连 trainer + cross-group)
  pixi run -e vllm vllm serve $ETHA_MODEL \
    --load-format dummy --trust-remote-code --enable-expert-parallel --enforce-eager \
    --tensor-parallel-size $TP --nnodes $INFER_NODES --node-rank 0 --master-addr $INFER_HEAD \
    --gpu-memory-utilization 0.85 --host 0.0.0.0 --port $ETHA_VLLM_PORT \
    --worker-extension-cls vllm_side.EthaWorkerExtension \
    --weight-transfer-config '{"backend": "nccl"}' &
  export ETHA_CROSS_HOST=$TRAIN_HEAD ETHA_TRAINER_HOST=$TRAIN_HEAD
  export TRAINER_WORLD=$(( TRAIN_NODES * 8 )) VLLM_TP=$TP
  pixi run -e vllm python examples/megatron_vllm/driver.py
fi
