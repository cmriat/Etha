#!/bin/bash
# Launch script for distributed tensor transfer with new TensorBus architecture
set -e
echo "🧹 Cleaning up old log and LMDB files..."

export ROOT_BASE=${HOME}/etha/prototyping
echo "ROOT_BASE: ${ROOT_BASE}"

export LMDB_ROOT=/tmp/dbs
export LOG_ROOT=${ROOT_BASE}/logs
rm -rf ${LMDB_ROOT}
rm -rf ${LOG_ROOT}
mkdir -p ${LMDB_ROOT}
mkdir -p ${LOG_ROOT}

export MODEL_ID=Qwen/Qwen3-0.6B

echo "🚀 Starting Agent processes (ranks 0-7)..."
pixi run -e dev torchrun --nproc_per_node=8 --master-port=39500 ${ROOT_BASE}/agent.py > ${LOG_ROOT}/agent.log 2>&1 &

# Wait for agents to be ready
echo "⏳ Waiting for agents to initialize..."
sleep 8

echo "🔥 Starting Training workers (ranks 0-3)..."
pixi run -e dev torchrun --nproc_per_node=4 --master-port=39501 \
    ${ROOT_BASE}/distributed_model_transfer/train.py > ${LOG_ROOT}/train.log 2>&1 &

echo "🔥 Starting Inference workers (ranks 0-3, connecting to agents 4-7)..."
AGENT_RANK_OFFSET=4 pixi run -e dev torchrun --nproc_per_node=4 --master-port=39502 \
    ${ROOT_BASE}/distributed_model_transfer/inference.py > ${LOG_ROOT}/inference.log 2>&1 &

echo ""
echo "✅ All processes started!"
echo "Press Ctrl+C to stop all processes..."
echo ""

# Wait for any process to exit or Ctrl+C
trap "echo 'Stopping all processes...'; pkill -f 'agent.py'; pkill -f 'train.py'; pkill -f 'inference.py'; exit" INT
wait