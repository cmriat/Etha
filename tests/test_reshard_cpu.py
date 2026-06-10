"""End-to-end reshard over gloo: plan, chunk, transfer, verify."""

import torch
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import Shard, Replicate, DeviceMesh, distribute_tensor

from etha import chunk_comm, get_m2m_map, split_fanout, m2m_to_chunks

WORLD = 4

CASES = {
    "p2p_disjoint_shard0_to_shard1": ([0, 1], (2,), (Shard(0),), [2, 3], (2,), (Shard(1),)),
    "broadcast_shard0_to_replicate": ([0, 1], (2,), (Shard(0),), [2, 3], (2,), (Replicate(),)),
    "local_identity": ([0, 1, 2, 3], (4,), (Shard(0),), [0, 1, 2, 3], (4,), (Shard(0),)),
    "overlap_2d_to_1d": (
        [0, 1, 2, 3],
        (2, 2),
        (Shard(0), Shard(1)),
        [0, 1, 2, 3],
        (4,),
        (Shard(0),),
    ),
    "fsdp_to_tp_fewer_ranks": ([0, 1, 2], (3,), (Shard(0),), [3], (1,), (Shard(1),)),
}


def _local(ref, mesh, placements):
    return distribute_tensor(ref, mesh, placements, src_data_rank=None).to_local()


def _run(rank, store, src_ranks, src_shape, src_pl, tgt_ranks, tgt_shape, tgt_pl, transfer_dtype, fanout=False):
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, init_method=f"file://{store}")
    ref = torch.arange(12 * 8 * 4, dtype=torch.float32).reshape(12, 8, 4)
    src_mesh_tensor = torch.tensor(src_ranks).reshape(src_shape)
    tgt_mesh_tensor = torch.tensor(tgt_ranks).reshape(tgt_shape)
    src_mesh = DeviceMesh("cpu", src_mesh_tensor)
    tgt_mesh = DeviceMesh("cpu", tgt_mesh_tensor)

    m2m = get_m2m_map(src_mesh_tensor, src_pl, tgt_mesh_tensor, tgt_pl)
    if fanout:
        m2m = split_fanout(m2m)

    source_tensor = _local(ref, src_mesh, src_pl).clone() if rank in src_ranks else None
    expected = _local(ref, tgt_mesh, tgt_pl) if rank in tgt_ranks else None
    if expected is not None and transfer_dtype is not None:
        expected = expected.to(transfer_dtype).to(expected.dtype)
    target_tensor = torch.zeros_like(expected) if expected is not None else None

    chunks = m2m_to_chunks(
        m2m, rank, source_tensor=source_tensor, target_tensor=target_tensor, transfer_dtype=transfer_dtype
    )
    chunk_comm(chunks)

    if expected is not None:
        torch.testing.assert_close(target_tensor, expected)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("name", CASES)
@pytest.mark.timeout(120)
def test_reshard(name, tmp_path):
    mp.spawn(_run, args=(tmp_path / "store", *CASES[name], None), nprocs=WORLD, join=True)


@pytest.mark.timeout(120)
def test_reshard_fanout(tmp_path):
    mp.spawn(
        _run,
        args=(tmp_path / "store", *CASES["broadcast_shard0_to_replicate"], None, True),
        nprocs=WORLD,
        join=True,
    )


@pytest.mark.timeout(120)
def test_reshard_wire_dtype(tmp_path):
    mp.spawn(
        _run,
        args=(tmp_path / "store", *CASES["p2p_disjoint_shard0_to_shard1"], torch.bfloat16),
        nprocs=WORLD,
        join=True,
    )
