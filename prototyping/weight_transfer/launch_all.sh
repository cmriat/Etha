#!/bin/bash

mkdir -p logs

echo "Starting all processes..."
rm prototyping/weight_transfer/dbs/* -rf
pixi run torchrun --nproc-per-node=8 --master-port=29500 prototyping/weight_transfer/middleware.py > prototyping/weight_transfer/logs/middle.log 2>&1 &
pixi run torchrun --nproc-per-node=4 --master-port=29501 prototyping/weight_transfer/train.py > prototyping/weight_transfer/logs/train.log 2>&1 &
pixi run torchrun --nproc-per-node=4 --master-port=29502 prototyping/weight_transfer/inference.py > prototyping/weight_transfer/logs/inference.log 2>&1 &

echo "All processes started! Check prototyping/weight_transfer/logs/ directory for output."