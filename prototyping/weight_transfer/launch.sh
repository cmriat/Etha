#!/bin/bash
# Launch script for weight transfer with new TensorBus architecture

set -e

echo "🧹 Cleaning up old log and LMDB files..."

rm -rf prototyping/weight_transfer/logs/* -rf
mkdir -p prototyping/weight_transfer/logs

rm -rf prototyping/weight_transfer/dbs/* -rf
mkdir -p prototyping/weight_transfer/dbs

echo "🚀 Starting Agent processes (ranks 0-7)..."
pixi run torchrun --nproc_per_node=8 --master-port=29500 prototyping/weight_transfer/agent.py > prototyping/weight_transfer/logs/agent.log 2>&1 &

# Wait for agents to be ready
echo "⏳ Waiting for agents to initialize..."
sleep 5

echo "🔥 Starting Training workers (ranks 0-3)..."
pixi run torchrun --nproc_per_node=4 --master-port=29501 prototyping/weight_transfer/train.py > prototyping/weight_transfer/logs/train.log 2>&1 &

echo "🔥 Starting Inference workers (ranks 0-3, connecting to agents 4-7)..."
AGENT_RANK_OFFSET=4 pixi run torchrun --nproc_per_node=4 --master-port=29502 prototyping/weight_transfer/inference.py > prototyping/weight_transfer/logs/inference.log 2>&1 &

echo ""
echo "✅ All processes started!"
echo "Press Ctrl+C to stop all processes..."
echo ""

# Wait for any process to exit or Ctrl+C
trap "echo 'Stopping all processes...'; pkill -f "python"; exit" INT
wait