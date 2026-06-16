"""NCCL reshard smoke test: 8 GPUs, fixed cases + fuzz, timing on rank 0.

torchrun --standalone --nproc_per_node=8 scripts/test_gpu.py
Default group is the DeviceMesh oracle; data plane runs on a store-built
cross-world NCCL communicator, mirroring production shape.
"""

import os
import math
import time
import random

import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard, Replicate, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from etha import chunk_comm, get_m2m_map, m2m_to_chunks, create_cross_group
from etha.planner import _tensor_ndim, _shard_counts

CASES = {
    "p2p_disjoint": ([0, 1, 2, 3], (4,), (Shard(0),), [4, 5, 6, 7], (4,), (Shard(1),)),
    "chain_broadcast": ([0, 1], (2,), (Shard(0),), [2, 3, 4, 5, 6, 7], (6,), (Replicate(),)),
    "overlap_2d": ([0, 1, 2, 3, 4, 5, 6, 7], (2, 4), (Shard(0), Shard(1)), [0, 1, 2, 3, 4, 5, 6, 7], (8,), (Shard(0),)),
    "ep_fsdp_strided": (
        [0, 1, 2, 3, 4, 5, 6, 7],
        (2, 4),
        (_StridedShard(0, split_factor=4), Shard(0)),
        [0, 1, 2, 3, 4, 5, 6, 7],
        (8,),
        (Shard(0),),
    ),
}

FUZZ_MESHES = [
    ((8,), list(range(8))),
    ((2, 4), list(range(8))),
    ((4, 2), list(range(8))),
    ((2, 2, 2), list(range(8))),
    ((4,), [0, 1, 2, 3]),
    ((4,), [4, 5, 6, 7]),
    ((2, 2), [2, 3, 6, 7]),
]


def _random_placements(rng, mesh_shape):
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


def run_case(rank, group, ref, src_ranks, src_shape, src_pl, tgt_ranks, tgt_shape, tgt_pl):
    src_mt = torch.tensor(src_ranks).reshape(src_shape)
    tgt_mt = torch.tensor(tgt_ranks).reshape(tgt_shape)
    m2m = get_m2m_map(src_mt, src_pl, tgt_mt, tgt_pl)
    src_mesh, tgt_mesh = DeviceMesh("cuda", src_mt), DeviceMesh("cuda", tgt_mt)

    def local(mesh, pl):
        return distribute_tensor(ref, mesh, pl, src_data_rank=None).to_local()

    source = local(src_mesh, src_pl).clone() if rank in src_ranks else None
    expected = local(tgt_mesh, tgt_pl) if rank in tgt_ranks else None
    target = torch.zeros_like(expected) if expected is not None else None
    chunks = m2m_to_chunks(m2m, rank, source_tensor=source, target_tensor=target)

    dist.barrier()
    t0 = time.perf_counter()
    chunk_comm(chunks, group=group)
    dt = time.perf_counter() - t0
    if expected is not None:
        torch.testing.assert_close(target, expected)
    return dt


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    group = create_cross_group(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]) + 1, rank, world)

    ref = torch.arange(2048 * 1024, dtype=torch.float32, device="cuda").reshape(2048, 1024)
    for name, case in CASES.items():
        dt = run_case(rank, group, ref, *case)
        if rank == 0:
            print(f"[case] {name}: {dt * 1e3:.1f} ms", flush=True)

    seed_t = torch.tensor([random.randrange(2**31)], device="cuda")
    dist.broadcast(seed_t, src=0)
    rng = random.Random(int(seed_t.item()))
    done = 0
    while done < 20:
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
        try:
            run_case(rank, group, ref, src_ranks, src_shape, src_pl, tgt_ranks, tgt_shape, tgt_pl)
        except NotImplementedError:
            continue
        done += 1
    if rank == 0:
        print(f"[fuzz] 20 rounds passed (seed={int(seed_t.item())})", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("ALL PASSED", flush=True)


if __name__ == "__main__":
    main()
