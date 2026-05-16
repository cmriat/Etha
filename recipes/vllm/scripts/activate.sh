#!/usr/bin/env bash
# Activation hook for the `vllm` conda package.
# flashinfer JIT-compiles fused_moe at runtime and needs CUDA headers + driver
# stubs visible to nvcc/clang; the vllm RPC layer needs cloudpickle enabled.

export CUDA_HOME="$CONDA_PREFIX"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
