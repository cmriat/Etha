#!/bin/bash
# Launch script for distributed model transfer with new TensorBus architecture
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

echo "🚀 Starting Agent processes (ranks 0-7) with torchrun..."
pixi run torchrun --nproc_per_node=8 --master-port=39500 ${PIXI_PROJECT_ROOT}/prototyping/agent.py > ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/logs/agent.log 2>&1 &
AGENT_PID=$!

# Wait for agents to be ready
echo "⏳ Waiting for agents to initialize..."
sleep 8

echo "🔥 Starting Training workers (ranks 0-3) with torchrun..."
TRAINING_STRATEGY=${TRAINING_STRATEGY:-"pure_mp"} pixi run torchrun --nproc_per_node=4 --master-port=39501 \
    ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/train.py > ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/logs/train.log 2>&1 &
TRAIN_PID=$!

echo "⚡ Starting Inference workers (ranks 0-3, connecting to agents 4-7) with torchrun..."
AGENT_RANK_OFFSET=4 CUDA_VISIBLE_DEVICES=4,5,6,7 INFERENCE_STRATEGY=${INFERENCE_STRATEGY:-"hybrid_dp_mp"} pixi run torchrun --nproc_per_node=4 --master-port=39502 \
    ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/inference.py > ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/logs/inference.log 2>&1 &
INFERENCE_PID=$!

echo ""
echo "✅ All processes started!"
echo "   - Agents (torchrun):    PID $AGENT_PID"
echo "   - Training (torchrun):  PID $TRAIN_PID"
echo "   - Inference (torchrun): PID $INFERENCE_PID"
echo ""
echo "📋 Monitor logs:"
echo "   tail -f ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/logs/agent.log"
echo "   tail -f ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/logs/train.log"
echo "   tail -f ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/logs/inference.log"
echo ""
echo "Press Ctrl+C to stop all processes..."
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping all processes..."
    pkill -f 'agent.py' 2>/dev/null || true
    pkill -f 'train.py' 2>/dev/null || true
    pkill -f 'inference.py' 2>/dev/null || true
    echo "✅ All processes stopped"
    exit
}

# Register cleanup on Ctrl+C
trap cleanup INT

# Wait for any process to exit
wait
