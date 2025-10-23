#!/bin/bash

echo "Stopping all processes..."
pkill -f "torchrun.*train.py"
pkill -f "torchrun.*middleware.py"
pkill -f "torchrun.*inference.py"
echo "All processes stopped!"