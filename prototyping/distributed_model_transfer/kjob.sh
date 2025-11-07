#!/bin/zsh
#SBATCH --job-name=etha_benchmark
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=1:00:00

# Job information

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Node ID: ${SLURM_NODEID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Total nodes: ${SLURM_NNODES}"
echo "============================================"

# NCCL configuration for distributed training
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_QPS_PER_CONNECTION="2"
export NCCL_IB_TIME_OUT="22"

# Distributed training environment variables
GROUP_PREFIX=$(echo "$SLURM_JOB_NODELIST" | sed 's/-[0-9]\..*//')
echo "GROUP_PREFIX: ${GROUP_PREFIX}"
ips=($(kubectl get pods -o wide | grep $GROUP_PREFIX | sort | awk '{print $6}'))
echo "IPs: ${ips[@]}"
echo "============================================"

# Setup environment
export ROOT_BASE=${HOME}/etha/prototyping
echo "ROOT_BASE: ${ROOT_BASE}"

export LMDB_ROOT=/tmp/dbs
export LOG_ROOT=${ROOT_BASE}/logs
mkdir -p ${LMDB_ROOT}
mkdir -p ${LOG_ROOT}

export NODE_RANK=${JOB_COMPLETION_INDEX}
export AGENT_NODE_OFFSET=1
export LOCAL_NODE_RANK=$((NODE_RANK % AGENT_NODE_OFFSET))
echo "NODE_RANK: ${NODE_RANK}, AGENT_NODE_OFFSET: ${AGENT_NODE_OFFSET} , LOCAL_NODE_RANK: ${LOCAL_NODE_RANK}"

export EXPECTED_WORLD_SIZE=8
export MODEL_ID=Qwen/Qwen3-0.6B


echo "============================================"
echo "Begin distributed model transfer..."

echo "🚀 Starting Agent processes on node ${NODE_RANK}..."
# Start agents with better error capture
pixi run -e dev --frozen --no-install \
    torchrun \
        --nnodes=${SLURM_NNODES} \
        --node-rank=${NODE_RANK} \
        --nproc-per-node=8 \
        --master-addr=${ips[1]} \
        --master-port=39500 \
        ${ROOT_BASE}/agent.py > ${LOG_ROOT}/agent_${NODE_RANK}.log 2>&1 &

echo "Node ${NODE_RANK}: Agent startup command completed"

echo "⏳ Waiting for agents to initialize..."
sleep 8

if [ ${NODE_RANK} -lt ${AGENT_NODE_OFFSET} ]; then
    # Node 0 to AGENT_NODE_OFFSET: Run training
    echo "🔥 Starting Training workers... at IP: ${ips[1]}"
    pixi run -e dev --frozen --no-install \
        torchrun \
            --nnodes=${AGENT_NODE_OFFSET} \
            --node-rank=${LOCAL_NODE_RANK} \
            --nproc-per-node=8 \
            --master-addr=${ips[1]} \
            --master-port=39501 \
            ${ROOT_BASE}/distributed_model_transfer/train.py > ${LOG_ROOT}/train.log 2>&1 &
else
    # Node AGENT_NODE_OFFSET to SLURM_NNODES: Run inference
    echo "🔥 Starting Inference workers... at IP: ${ips[$((1 + AGENT_NODE_OFFSET))]}"
    AGENT_RANK_OFFSET=8 pixi run -e dev --frozen --no-install \
        torchrun \
            --nnodes=$((SLURM_NNODES - AGENT_NODE_OFFSET)) \
            --node-rank=${LOCAL_NODE_RANK} \
            --nproc-per-node=8 \
            --master-addr=${ips[$((1 + AGENT_NODE_OFFSET))]} \
            --master-port=39502 \
            ${ROOT_BASE}/distributed_model_transfer/inference.py > ${LOG_ROOT}/inference.log 2>&1 &
fi



# Monitor logs for completion indicators
while true; do
    # Check if all processes are still running
    if ! pgrep -f "torchrun.*agent.py" > /dev/null && \
       ! pgrep -f "torchrun.*train.py" > /dev/null && \
       ! pgrep -f "torchrun.*inference.py" > /dev/null; then
        echo "============================================"
        echo "All distributed model transfer processes completed"
        echo "============================================"
        break
    fi

    # Sleep for a bit before checking again
    sleep 30
    echo "$(date): Processes still running..."
done