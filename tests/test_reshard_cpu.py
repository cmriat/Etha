"""End-to-end reshard: plan, chunk, transfer, verify.

The default group (gloo) exists only for the DeviceMesh oracle; every
``chunk_comm`` runs over a store-built cross-world communicator, mirroring the
production shape (engines own the default group, etha owns its own).
"""

import math
import random
import socket

import torch
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import Shard, Replicate, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from etha import chunk_comm, get_m2m_map, m2m_to_chunks, create_cross_group
from etha.planner import _tensor_ndim, _shard_counts

WORLD = 4


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
    "nested_double_shard_same_dim": (
        [0, 1, 2, 3],
        (2, 2),
        (Shard(0), Shard(0)),
        [0, 1, 2, 3],
        (4,),
        (Shard(1),),
    ),
    "ep_fsdp_strided_shard": (
        [0, 1, 2, 3],
        (2, 2),
        (_StridedShard(0, split_factor=2), Shard(0)),
        [0, 1, 2, 3],
        (4,),
        (Shard(0),),
    ),
}


def _local(ref, mesh, placements):
    return distribute_tensor(ref, mesh, placements, src_data_rank=None).to_local()


def _run(rank, store, port, src_ranks, src_shape, src_pl, tgt_ranks, tgt_shape, tgt_pl, transfer_dtype):
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, init_method=f"file://{store}")
    group = create_cross_group("127.0.0.1", port, rank, WORLD, backend="gloo")
    ref = torch.arange(12 * 8 * 4, dtype=torch.float32).reshape(12, 8, 4)
    src_mesh_tensor = torch.tensor(src_ranks).reshape(src_shape)
    tgt_mesh_tensor = torch.tensor(tgt_ranks).reshape(tgt_shape)
    src_mesh = DeviceMesh("cpu", src_mesh_tensor)
    tgt_mesh = DeviceMesh("cpu", tgt_mesh_tensor)

    m2m = get_m2m_map(src_mesh_tensor, src_pl, tgt_mesh_tensor, tgt_pl)

    source_tensor = _local(ref, src_mesh, src_pl).clone() if rank in src_ranks else None
    expected = _local(ref, tgt_mesh, tgt_pl) if rank in tgt_ranks else None
    if expected is not None and transfer_dtype is not None:
        expected = expected.to(transfer_dtype).to(expected.dtype)
    target_tensor = torch.zeros_like(expected) if expected is not None else None

    chunks = m2m_to_chunks(
        m2m, rank, source_tensor=source_tensor, target_tensor=target_tensor, transfer_dtype=transfer_dtype
    )
    chunk_comm(chunks, group=group)

    if expected is not None:
        torch.testing.assert_close(target_tensor, expected)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("name", CASES)
@pytest.mark.timeout(120)
def test_reshard(name, tmp_path):
    mp.spawn(_run, args=(tmp_path / "store", _free_port(), *CASES[name], None), nprocs=WORLD, join=True)


FUZZ_MESHES = [
    ((4,), [0, 1, 2, 3]),
    ((2, 2), [0, 1, 2, 3]),
    ((1, 4), [0, 1, 2, 3]),
    ((2, 1, 2), [0, 1, 2, 3]),
    ((2,), [0, 1]),
    ((2,), [2, 3]),
]


def _random_placements(rng, mesh_shape):
    """``split_factor`` is derived from the structure, the way FSDP2 fills it."""
    kinds = []
    for _ in mesh_shape:
        roll = rng.random()
        if roll < 0.45:
            kinds.append(("shard", rng.randrange(2)))
        elif roll < 0.6:
            kinds.append(("strided", rng.randrange(2)))
        else:
            kinds.append(("replicate", None))
    out = []
    for i, (kind, dim) in enumerate(kinds):
        if kind == "replicate":
            out.append(Replicate())
        elif kind == "shard":
            out.append(Shard(dim))
        else:
            inner = math.prod(
                mesh_shape[j] for j, (k, d) in enumerate(kinds) if j > i and k != "replicate" and d == dim
            )
            out.append(_StridedShard(dim, split_factor=inner))
    return tuple(out)


def _run_fuzz(rank, store, port, seed, rounds):
    """Same seed on every rank: all draw the identical case sequence."""
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, init_method=f"file://{store}")
    group = create_cross_group("127.0.0.1", port, rank, WORLD, backend="gloo")
    rng = random.Random(seed)
    ref = torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16)
    done = 0
    while done < rounds:
        src_shape, src_ranks = FUZZ_MESHES[rng.randrange(len(FUZZ_MESHES))]
        tgt_shape, tgt_ranks = FUZZ_MESHES[rng.randrange(len(FUZZ_MESHES))]
        src_pl = _random_placements(rng, src_shape)
        tgt_pl = _random_placements(rng, tgt_shape)

        ndim = max(_tensor_ndim(src_pl), _tensor_ndim(tgt_pl))
        middle = [
            math.lcm(s, t)
            for s, t in zip(_shard_counts(src_shape, src_pl, ndim), _shard_counts(tgt_shape, tgt_pl, ndim), strict=True)
        ]
        if any(ref.shape[d] % middle[d] for d in range(ndim)):
            continue

        src_mt = torch.tensor(src_ranks).reshape(src_shape)
        tgt_mt = torch.tensor(tgt_ranks).reshape(tgt_shape)
        try:
            m2m = get_m2m_map(src_mt, src_pl, tgt_mt, tgt_pl)
        except NotImplementedError:
            continue  # planner rejects the combo on every rank alike; sample again
        src_mesh, tgt_mesh = DeviceMesh("cpu", src_mt), DeviceMesh("cpu", tgt_mt)
        source_tensor = _local(ref, src_mesh, src_pl).clone() if rank in src_ranks else None
        expected = _local(ref, tgt_mesh, tgt_pl) if rank in tgt_ranks else None
        target_tensor = torch.zeros_like(expected) if expected is not None else None
        chunks = m2m_to_chunks(m2m, rank, source_tensor=source_tensor, target_tensor=target_tensor)
        chunk_comm(chunks, group=group)
        if expected is not None:
            torch.testing.assert_close(target_tensor, expected, msg=f"seed={seed} case {done}: {src_pl} -> {tgt_pl}")
        done += 1
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.timeout(300)
def test_reshard_fuzz(tmp_path):
    seed = random.randrange(2**32)
    mp.spawn(_run_fuzz, args=(tmp_path / "store", _free_port(), seed, 10), nprocs=WORLD, join=True)




@pytest.mark.timeout(120)
def test_reshard_wire_dtype(tmp_path):
    mp.spawn(
        _run,
        args=(tmp_path / "store", _free_port(), *CASES["p2p_disjoint_shard0_to_shard1"], torch.bfloat16),
        nprocs=WORLD,
        join=True,
    )


def _run_streaming(rank, store, port):
    """Deferred dst allocation + per-weight completion hand-off."""
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, init_method=f"file://{store}")
    group = create_cross_group("127.0.0.1", port, rank, WORLD, backend="gloo")
    src_ranks, src_shape, src_pl, tgt_ranks, tgt_shape, tgt_pl = CASES["broadcast_shard0_to_replicate"]
    ref = torch.arange(12 * 8 * 4, dtype=torch.float32).reshape(12, 8, 4)
    src_mt = torch.tensor(src_ranks).reshape(src_shape)
    tgt_mt = torch.tensor(tgt_ranks).reshape(tgt_shape)
    m2m = get_m2m_map(src_mt, src_pl, tgt_mt, tgt_pl)
    src_mesh, tgt_mesh = DeviceMesh("cpu", src_mt), DeviceMesh("cpu", tgt_mt)

    source = _local(ref, src_mesh, src_pl).clone() if rank in src_ranks else None
    expected = _local(ref, tgt_mesh, tgt_pl) if rank in tgt_ranks else None
    if rank in src_ranks:
        chunks = m2m_to_chunks(m2m, rank, source_tensor=source)
    else:
        chunks = m2m_to_chunks(m2m, rank, target_shape=tuple(expected.shape), transfer_dtype=torch.float32)
    for c in chunks:
        c.weight = "w0"

    received = {}
    chunk_comm(
        chunks,
        group=group,
        target_alloc=lambda n: torch.zeros(tuple(expected.shape)),
        on_complete=lambda n, buf: received.__setitem__(n, buf),
    )
    if expected is not None and rank not in src_ranks:
        torch.testing.assert_close(received["w0"], expected)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_reshard_streaming(tmp_path):
    mp.spawn(_run_streaming, args=(tmp_path / "store", _free_port()), nprocs=WORLD, join=True)
