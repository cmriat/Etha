#!/bin/bash
# Launch script for distributed tensor transfer with new TensorBus architecture
set -e
echo "🧹 Cleaning up old log and LMDB files..."

rm -rf ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/logs/*
mkdir -p ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/logs

rm -rf /tmp/dbs/*
mkdir -p /tmp/dbs

echo "🚀 Starting Agent processes (ranks 0-7)..."
torchrun --nproc_per_node=8 --master-port=39500 ${PIXI_PROJECT_ROOT}/prototyping/agent.py > ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/logs/agent.log 2>&1 &

# Wait for agents to be ready
echo "⏳ Waiting for agents to initialize..."
sleep 8

echo "🔥 Starting Training workers (ranks 0-3)..."
TRAINING_STRATEGY=${TRAINING_STRATEGY:-"pure_mp"} torchrun --nproc_per_node=4 --master-port=39501 \
    ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/train.py > ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/logs/train.log 2>&1 &

echo "🔥 Starting Inference workers (ranks 0-3, connecting to agents 4-7)..."
AGENT_RANK_OFFSET=4 INFERENCE_STRATEGY=${INFERENCE_STRATEGY:-"hybrid_dp_mp"} torchrun --nproc_per_node=4 --master-port=39502 \
    ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/inference.py > ${PIXI_PROJECT_ROOT}/prototyping/distributed_model_transfer/logs/inference.log 2>&1 &

echo ""
echo "✅ All processes started!"
echo "Press Ctrl+C to stop all processes..."
echo ""

# Wait for any process to exit or Ctrl+C
trap "echo 'Stopping all processes...'; pkill -f 'agent.py'; pkill -f 'train.py'; pkill -f 'inference.py'; exit" INT
wait