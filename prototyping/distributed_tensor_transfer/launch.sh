#!/bin/bash
# Ray-based launch script for distributed tensor transfer
# Uses Ray actors for inference workers to reproduce NCCL device ID mapping issues
set -e

echo "🧹 Cleaning up old log and LMDB files..."

export ROOT_BASE=${PIXI_PROJECT_ROOT}/prototyping
echo "ROOT_BASE: ${ROOT_BASE}"

export LMDB_ROOT=/tmp/dbs
export LOG_ROOT=${ROOT_BASE}/distributed_tensor_transfer/logs
rm -rf ${LMDB_ROOT}
rm -rf ${LOG_ROOT}
mkdir -p ${LMDB_ROOT}
mkdir -p ${LOG_ROOT}

echo "🚀 Starting Agent processes (ranks 0-7) with torchrun..."
pixi run torchrun --nproc_per_node=8 --master-port=39500 ${ROOT_BASE}/agent.py > ${LOG_ROOT}/agent.log 2>&1 &
AGENT_PID=$!

# Wait for agents to be ready
echo "⏳ Waiting for agents to initialize..."
sleep 8

echo "🔥 Starting Training workers (ranks 0-3) with torchrun..."
pixi run torchrun --nproc_per_node=4 --master-port=39501 \
    ${ROOT_BASE}/distributed_tensor_transfer/train.py > ${LOG_ROOT}/train.log 2>&1 &
TRAIN_PID=$!

echo "⚡ Starting Inference workers (ranks 0-3) with RAY..."
echo "   (Connecting to agents 4-7, expecting NCCL device ID issues)"
AGENT_RANK_OFFSET=4 CUDA_VISIBLE_DEVICES=4,5,6,7 pixi run python \
    ${ROOT_BASE}/distributed_tensor_transfer/ray_inference.py > ${LOG_ROOT}/inference.log 2>&1 &
INFERENCE_PID=$!

echo ""
echo "✅ All processes started!"
echo "   - Agents (torchrun):    PID $AGENT_PID"
echo "   - Training (torchrun):  PID $TRAIN_PID"
echo "   - Inference (Ray):      PID $INFERENCE_PID"
echo ""
echo "📋 Monitor logs:"
echo "   tail -f ${LOG_ROOT}/agent.log"
echo "   tail -f ${LOG_ROOT}/train.log"
echo "   tail -f ${LOG_ROOT}/inference.log"
echo ""
echo "Press Ctrl+C to stop all processes..."
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping all processes..."
    pkill -f 'agent.py' 2>/dev/null || true
    pkill -f 'train.py' 2>/dev/null || true
    pkill -f 'ray_inference.py' 2>/dev/null || true
    pkill -f 'ray::' 2>/dev/null || true  # Kill Ray workers
    echo "✅ All processes stopped"
    exit
}

# Register cleanup on Ctrl+C
trap cleanup INT

# Wait for any process to exit
wait
