"""Minimal shared config for the distributed model transfer integration test."""

import os

import torch
from torch.distributed.tensor.placement_types import Shard, Replicate

PAIR_NAME = "distributed_weights"
EXPECTED_WORLD_SIZE = int(os.environ.get("EXPECTED_WORLD_SIZE", 4))

MESH_CONFIGS = {
    "hybrid_dp_mp": ((2, EXPECTED_WORLD_SIZE // 2), (Replicate(), Shard(0))),
    "pure_mp": ((EXPECTED_WORLD_SIZE,), (Shard(0),)),
}

STORE_HOST = os.environ.get("MASTER_ADDR", "localhost")
STORE_PORT = int(os.environ.get("ETHA_STORE_PORT", 40001))
STORE_BACKEND = "tcp"
LMDB_ROOT = os.environ.get("LMDB_ROOT", "/tmp/dbs")


def get_queue_state_paths(rank: int) -> tuple[str, str]:
    return (f"{LMDB_ROOT}/{rank}_command.lmdb", f"{LMDB_ROOT}/{rank}_state.lmdb")


def get_model_dtype_from_env(env_var: str = "MODEL_DTYPE", default: str = "bfloat16") -> torch.dtype:
    value = os.environ.get(env_var, default)
    dtype = getattr(torch, value, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype '{value}'")
    return dtype
