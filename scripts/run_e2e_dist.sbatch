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
export PYTHONPATH=$PWD/examples/megatron_vllm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_SERVER_DEV_MODE=1
export TRITON_CACHE_DIR=/tmp/triton-cache TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache VLLM_CACHE_ROOT=/tmp/vllm-cache

# ── 节点发现(kubectl pod IP,按 pod 名排序)──
GROUP_PREFIX=$(echo "$SLURM_JOB_NODELIST" | sed 's/-[0-9]\..*//')

# 端口按 job 唯一化:hostNetwork 下端口按物理机共享,刚删的 job 进程释放有延迟、
# 会占住端口。每 job 一个 512 宽的块(块间不重叠;zmq 单 job 就要占 base+0..31+cross,
# 故块必须远宽于单端口)。GROUP_PREFIX 每 job 唯一且全 pod 共享 → cksum 选块。
BLOCK=$(( 10000 + $(echo "$GROUP_PREFIX" | cksum | cut -d' ' -f1) % 40 * 512 ))
MASTER_PORT=$BLOCK                             # torchrun c10d
export ETHA_HTTP_PORT=$(( BLOCK + 10 ))        # rpc:zmq base = +100 (+rank 0..31), cross = +200
export ETHA_VLLM_PORT=$(( BLOCK + 300 ))       # vLLM API server (--port)
# 不设 VLLM_PORT:vLLM 内部 MQ 用 bind(("",0)) 由 OS 分配、天然分散;设固定 base
# 反而让同机 8 worker 全挤同一端口(get_open_port 的 check-close-rebind 竞态)而互撞。
for i in {1..240}; do  # 等满 8min:GPU 紧张时部分 pod 调度慢,给它们排上的时间
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
    --master-addr $TRAIN_HEAD --master-port $MASTER_PORT --nproc_per_node 8 \
    examples/megatron_vllm/trainer_server.py
else
  # ── inference 节点:vLLM(TP 跨节点 / DP-attention)──
  IRANK=$(( RANK - TRAIN_NODES ))
  GPUS=$(( INFER_NODES * 8 ))
  QUANT=${INFER_QUANT:+--quantization $INFER_QUANT}
  MAXLEN=${ETHA_MAX_LEN:+--max-model-len $ETHA_MAX_LEN}
  BATCHED=${ETHA_MAX_BATCHED:+--max-num-batched-tokens $ETHA_MAX_BATCHED}  # 砍 profile run activation 峰值
  UTIL=${ETHA_GPU_UTIL:-0.85}
  GDN=${ETHA_GDN_BACKEND:+--gdn-prefill-backend $ETHA_GDN_BACKEND}  # GatedDeltaNet(Qwen3.5/Next):triton 绕开 flashinfer 硬依赖(sm_90 默认走 flashinfer 但未装)
  MMLIMIT=${ETHA_NO_VISION:+--limit-mm-per-prompt {\"image\":0,\"video\":0}}  # profiling 跳过 vision encoder profile run(transfer 只用 language_model,text-only 推理不受影响),省 init ~60s
  if [ -n "$ETHA_DP_ATTN" ]; then
    # DP-attention:dp=GPUS, tp=1, EP;attention/shared replicate、experts EP → 转置消失
    DP_PORT=$(( BLOCK + 400 ))
    PAR="--tensor-parallel-size 1 --data-parallel-size $GPUS --data-parallel-size-local 8 --data-parallel-address $INFER_HEAD --data-parallel-rpc-port $DP_PORT"
    SEC="--data-parallel-start-rank $(( IRANK * 8 ))"
    APISRV="--api-server-count 1"  # DP 默认 api_server_count=dp_size=32,head 节点 32 个 API server 抢端口致 ApiServer 死;只要 1 个给 driver
  else
    PAR="--tensor-parallel-size $GPUS --nnodes $INFER_NODES --master-addr $INFER_HEAD"
    SEC="--node-rank $IRANK"
    HEAD="--node-rank 0"
  fi
  if [ "$IRANK" -ne 0 ]; then
    # 非 head:headless worker 前台跑(无 wait,kjobctl 不支持 wait)
    exec pixi run -e vllm vllm serve $ETHA_MODEL \
      --load-format ${ETHA_LOAD_FORMAT:-dummy} --trust-remote-code --enable-expert-parallel --enforce-eager $QUANT $MAXLEN $BATCHED $GDN $MMLIMIT \
      $PAR $SEC --gpu-memory-utilization $UTIL --headless \
      --worker-extension-cls vllm_side.EthaWorkerExtension \
      --weight-transfer-config '{"backend": "nccl"}'
  fi
  # infer head:serve 后台 + driver(本地连 vLLM,经 train head 连 trainer + cross-group)
  pixi run -e vllm vllm serve $ETHA_MODEL \
    --load-format ${ETHA_LOAD_FORMAT:-dummy} --trust-remote-code --enable-expert-parallel --enforce-eager $QUANT $MAXLEN $BATCHED $GDN $MMLIMIT \
    $PAR ${HEAD:-} ${APISRV:-} --gpu-memory-utilization $UTIL --host 0.0.0.0 --port $ETHA_VLLM_PORT \
    --worker-extension-cls vllm_side.EthaWorkerExtension \
    --weight-transfer-config '{"backend": "nccl"}' &
  export ETHA_CROSS_HOST=$TRAIN_HEAD ETHA_TRAINER_HOST=$TRAIN_HEAD
  export TRAINER_WORLD=$(( TRAIN_NODES * 8 )) VLLM_TP=$GPUS
  pixi run -e vllm python examples/megatron_vllm/driver.py
fi
