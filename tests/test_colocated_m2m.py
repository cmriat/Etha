"""Colocated mesh-to-mesh communication: both meshes span the same ranks.

One agent hosting both endpoints of a colocated pair registers two meshes
that cover the same physical ranks. This file exercises the comm-layer cases
the disjoint-mesh tests cannot reach:

- get_m2m_map's trace phase where the trace target rank is the rank itself
  (the trace used to isend to itself and hang);
- routes whose destination lands on the source rank — LOCAL chunks between
  two *different* tensors on one rank (the source and target registrations);
- the reverse direction (Replicate, Shard(1)) -> (Shard(0),) whose routes are
  P2P with the single destination equal to the source (refined to LOCAL).

Also pins the master-stability contract: a transfer out of an FP32 master
casts on the wire and never mutates the source.
"""

import os
import socket
import logging

import torch
import pytest
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Shard, Replicate

from etha.comm import (
    bucket_comm,
    get_m2m_map,
    m2m_to_chunks,
    chunk_to_bucket_ops,
)
from etha.comm.ir import Transport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_colocated_worker(rank: int, world_size: int, device: str) -> None:
    dist.init_process_group(backend="gloo" if device == "cpu" else "nccl", rank=rank, world_size=world_size)
    torch.manual_seed(0)

    train_mesh = DeviceMesh(device, torch.arange(world_size))  # (4,) Shard(0)
    infer_mesh = DeviceMesh(device, torch.arange(world_size).view(world_size // 2, 2))  # (2,2) (Rep, Shard(1))
    train_specs = (Shard(0),)
    infer_specs = (Replicate(), Shard(1))

    shape = (8, 4)
    master = torch.randn(shape, device=device)  # seeded: identical on every rank
    train_dt = distribute_tensor(master, train_mesh, train_specs)
    train_local = train_dt.to_local()
    train_snapshot = train_local.clone()

    # --- forward, same dtype: one call carries source AND target registration
    infer_dt = distribute_tensor(torch.zeros(shape, device=device), infer_mesh, infer_specs)
    infer_local = infer_dt.to_local()

    fwd = get_m2m_map(train_mesh, train_specs, infer_mesh, infer_specs, group=dist.group.WORLD, device=device)
    # Reversed call on colocated meshes: the old trace self-send hung here.
    rev = get_m2m_map(infer_mesh, infer_specs, train_mesh, train_specs, group=dist.group.WORLD, device=device)
    logger.info(f"[rank={rank}] generated both colocated maps without deadlock")

    chunks = m2m_to_chunks(fwd, rank=rank, source_tensor=train_local, target_tensor=infer_local)
    local_chunks = [c for c in chunks if c.transport is Transport.LOCAL]
    assert local_chunks, f"rank {rank}: colocated forward direction must produce LOCAL chunks"
    for chunk in local_chunks:
        assert chunk.is_source and chunk.is_target
        # LOCAL on colocated meshes reads a *different* tensor than it writes
        assert chunk.src_tensor is not None and chunk.src_tensor is not chunk.tensor

    # bucket_size=1 keeps every bucket single-chunk (CPU-safe: no CUDA events)
    bucket_comm(buckets=chunk_to_bucket_ops(chunks=chunks, bucket_size=1))

    assert torch.equal(infer_dt.full_tensor(), master), f"rank {rank}: forward reshard mismatch"
    assert torch.equal(train_local, train_snapshot), f"rank {rank}: forward mutated the source"

    # --- reverse: (Replicate, Shard(1)) -> (Shard(0),) refines self-P2P to LOCAL
    back_dt = distribute_tensor(torch.zeros(shape, device=device), train_mesh, train_specs)
    back_local = back_dt.to_local()
    rev_chunks = m2m_to_chunks(rev, rank=rank, source_tensor=infer_local, target_tensor=back_local)

    local_counts = [None] * world_size
    dist.all_gather_object(local_counts, sum(1 for c in rev_chunks if c.transport is Transport.LOCAL))
    assert sum(local_counts) >= 1, f"reverse direction produced no LOCAL chunks anywhere: {local_counts}"

    bucket_comm(buckets=chunk_to_bucket_ops(chunks=rev_chunks, bucket_size=1))
    assert torch.equal(back_dt.full_tensor(), master), f"rank {rank}: reverse reshard mismatch"

    # --- forward with cast: FP32 master -> BF16 target, wire dtype = min itemsize
    infer16_dt = distribute_tensor(torch.zeros(shape, dtype=torch.bfloat16, device=device), infer_mesh, infer_specs)
    infer16_local = infer16_dt.to_local()
    cast_chunks = m2m_to_chunks(
        fwd, rank=rank, source_tensor=train_local, target_tensor=infer16_local, transfer_dtype=torch.bfloat16
    )
    bucket_comm(buckets=chunk_to_bucket_ops(chunks=cast_chunks, bucket_size=1))

    assert torch.equal(infer16_dt.full_tensor(), master.to(torch.bfloat16)), f"rank {rank}: cast reshard mismatch"
    assert torch.equal(train_local, train_snapshot), f"rank {rank}: cast transfer mutated the FP32 master"

    dist.destroy_process_group()


@pytest.mark.timeout(600)
def test_colocated_m2m_cpu():
    """Colocated forward/reverse resharding without deadlock, bitwise-correct."""
    world_size = 4
    device = "cpu"

    os.environ["MASTER_ADDR"] = "localhost"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])

    try:
        torch.multiprocessing.spawn(
            run_colocated_worker,
            args=(world_size, device),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        pytest.fail(f"Colocated m2m test failed: {e}")
