"""Reshard test for the trainer↔vLLM MoE pattern.

Covers source `(1, 2, 2)` with `(Replicate, Shard(0), Shard(0))` →
target `(4,)` with `(Shard(0),)`, plus a few neighbouring placements.
The trainer's MoE mesh in `examples/vllm_weight_sync` uses 3D
`(dp_replicate=1, dp_shard=2, ep=2)` and vLLM's uses 1D `(ep=4,)`;
both partition E into 4 contiguous chunks row-major, so the reshard
should be a 1-to-1 rank match. This test runs that reshard (and the
gate/up slice-view variant) on CPU and asserts the received tensor
matches the source globally.
"""

import os
import math
import socket
import logging

import torch
import pytest
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed._tensor import Shard, Replicate, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Partial

from etha.comm import chunk_comm, bucket_comm, get_m2m_map, map_to_chunk_ops, chunk_to_bucket_ops
from etha.tensor_bus.agent import _create_partial_groups

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _local_shape(
    tensor_shape: tuple[int, ...],
    mesh_shape: tuple[int, ...],
    placements: tuple,
) -> tuple[int, ...]:
    """Per-rank local shape after Shard partitioning."""
    local = list(tensor_shape)
    for mesh_dim, p in enumerate(placements):
        if isinstance(p, Shard):
            local[p.dim] //= mesh_shape[mesh_dim]
    return tuple(local)


def _run(
    rank: int,
    world_size: int,
    source_mesh_shape: tuple[int, ...],
    source_placements: tuple,
    target_mesh_shape: tuple[int, ...],
    target_placements: tuple,
    tensor_shape: tuple[int, ...],
    device: str,
    slice_view: bool = False,
    comm_method: str = "chunk",
) -> None:
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    source_ws = math.prod(source_mesh_shape)
    target_ws = math.prod(target_mesh_shape)
    assert world_size == source_ws + target_ws

    source_mesh = DeviceMesh(device, torch.arange(source_ws).view(source_mesh_shape))
    target_mesh = DeviceMesh(
        device,
        torch.arange(source_ws, source_ws + target_ws).view(target_mesh_shape),
    )

    torch.manual_seed(0)
    src_global = torch.randn(tensor_shape, device=device)
    tgt_global = torch.randn(tensor_shape, device=device)

    is_source = rank < source_ws
    has_partial_source = any(isinstance(p, Partial) for p in source_placements)
    if has_partial_source:
        # Partial cannot be the entry point to distribute_tensor; build per-rank
        # random local contributions and wrap with DTensor.from_local. The
        # logical (post-reduce) tensor is the cross-PG ground truth.
        torch.manual_seed(1000 + rank)
        if is_source:
            src_local_shape = _local_shape(tensor_shape, source_mesh_shape, source_placements)
            src_local = torch.randn(src_local_shape, device=device)
            src_dt = DTensor.from_local(src_local, source_mesh, list(source_placements))
            tgt_dt, tgt_local = None, None
        else:
            tgt_dt = distribute_tensor(tgt_global, target_mesh, list(target_placements))
            tgt_local = tgt_dt.to_local()
            src_dt, src_local = None, None
        # Compute logical (post-reduce) tensor on source side, send to every target
        # rank so each one has a ground truth to compare against after the reshard.
        if is_source:
            full_logical = src_dt.redistribute(source_mesh, [Replicate()] * len(source_placements)).to_local()
        else:
            full_logical = torch.empty(tensor_shape, device=device)
        if is_source and rank == 0:
            for tr in range(source_ws, world_size):
                dist.send(full_logical, dst=tr)
        if not is_source:
            dist.recv(full_logical, src=0)
    elif slice_view:
        wide_shape = list(tensor_shape)
        wide_shape[1] *= 2
        torch.manual_seed(42)
        src_wide_full = torch.randn(wide_shape, device=device)
        torch.manual_seed(99)
        tgt_wide_full = torch.randn(wide_shape, device=device)
        if is_source:
            src_dt = distribute_tensor(src_wide_full, source_mesh, list(source_placements))
            src_local = src_dt.to_local()[:, : tensor_shape[1], :]
            tgt_dt, tgt_local = None, None
        else:
            tgt_dt = distribute_tensor(tgt_wide_full, target_mesh, list(target_placements))
            tgt_local = tgt_dt.to_local()[:, : tensor_shape[1], :]
            src_dt, src_local = None, None
    else:
        if is_source:
            src_dt = distribute_tensor(src_global, source_mesh, list(source_placements))
            src_local = src_dt.to_local()
            tgt_dt, tgt_local = None, None
        else:
            tgt_dt = distribute_tensor(tgt_global, target_mesh, list(target_placements))
            tgt_local = tgt_dt.to_local()
            src_dt, src_local = None, None

    m2m_map, src_slicers, tgt_slicers, partial_red = get_m2m_map(
        source_mesh=source_mesh,
        source_placements=list(source_placements),
        target_mesh=target_mesh,
        target_placements=list(target_placements),
        group=dist.group.WORLD,
        device=device,
    )

    # Source-side Partial sub-groups (collective on WORLD; non-members still call).
    source_ranks_list = list(range(source_ws))
    source_group = dist.new_group(ranks=source_ranks_list)
    source_partial_groups = _create_partial_groups(
        mesh_tensor=source_mesh.mesh,
        partial_reductions=partial_red,
        this_rank=rank,
        full_source_ranks=source_ranks_list,
        full_source_group=source_group,
    )

    chunks = map_to_chunk_ops(
        m2m_map=m2m_map,
        rank=rank,
        source_num_slicers=src_slicers,
        target_num_slicers=tgt_slicers,
        source_tensor=src_local,
        target_tensor=tgt_local,
        source_partial_groups=source_partial_groups or None,
    )
    # Snapshot the source tensor so we can verify the Partial reduce path did
    # not mutate the worker's data via an aliasing in-place all_reduce.
    src_local_snapshot = (
        src_local.detach().clone() if (is_source and has_partial_source and src_local is not None) else None
    )
    match comm_method:
        case "chunk":
            chunk_comm(chunks=chunks)
        case "bucket":
            buckets = chunk_to_bucket_ops(chunks=chunks, bucket_size=256 * 1024 * 1024)
            bucket_comm(buckets=buckets)
        case _:
            raise ValueError(f"unknown comm_method: {comm_method}")

    if is_source and src_local_snapshot is not None:
        assert torch.equal(src_local, src_local_snapshot), (
            f"source tensor was mutated by Partial reduce on rank {rank}: "
            f"max diff {(src_local - src_local_snapshot).abs().max().item()}"
        )

    if not is_source:
        if has_partial_source:
            full = tgt_dt.full_tensor()
            assert torch.allclose(full, full_logical, rtol=1e-4, atol=1e-4), (
                f"partial reshard mismatch on rank {rank}: max diff {(full - full_logical).abs().max().item()}"
            )
        elif slice_view:
            tgt_full = tgt_dt.full_tensor()
            half = tensor_shape[1]
            got = tgt_full[:, :half, :]
            expected = src_wide_full[:, :half, :]
            unchanged_got = tgt_full[:, half:, :]
            unchanged_expected = tgt_wide_full[:, half:, :]
            assert torch.allclose(got, expected), (
                f"slice-view reshard mismatch on rank {rank}: "
                f"max diff in transferred half = {(got - expected).abs().max().item()}"
            )
            assert torch.allclose(unchanged_got, unchanged_expected), (
                f"slice-view comm clobbered the second half on rank {rank}: "
                f"max diff in untouched half = {(unchanged_got - unchanged_expected).abs().max().item()}"
            )
        else:
            full = tgt_dt.full_tensor()
            assert torch.allclose(full, src_global), (
                f"reshard mismatch on rank {rank}: max diff {(full - src_global).abs().max().item()}"
            )

    dist.destroy_process_group()


def _run_partial_mixed_dtype(
    rank: int,
    world_size: int,
    source_ws: int,
    target_ws: int,
    tensor_shape: tuple[int, ...],
    device: str,
    comm_method: str,
) -> None:
    """fp32 Partial source → bf16 Replicate target.

    Regression for the P1: source-side all-reduce must happen in the source
    (fp32) dtype before the cast to ``transfer_dtype`` (bf16), matching
    DTensor's ``Partial → Replicate`` semantics. Inputs are scaled large so
    the alternative (cast-first then bf16 all-reduce) diverges enough from
    the fp32-reduce-then-cast reference to exceed the allclose tolerance.
    """
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    source_mesh_shape = (source_ws,)
    target_mesh_shape = (target_ws,)
    source_placements = (Partial(),)
    target_placements = (Replicate(),)
    transfer_dtype = torch.bfloat16

    source_mesh = DeviceMesh(device, torch.arange(source_ws).view(source_mesh_shape))
    target_mesh = DeviceMesh(
        device,
        torch.arange(source_ws, source_ws + target_ws).view(target_mesh_shape),
    )

    is_source = rank < source_ws
    torch.manual_seed(1000 + rank)
    if is_source:
        # Large magnitude so bf16-side reduction loses precision distinctly
        # from fp32-side reduction.
        src_local = torch.randn(tensor_shape, dtype=torch.float32, device=device) * 1e3
        src_dt = DTensor.from_local(src_local, source_mesh, list(source_placements))
        full_logical_fp32 = src_dt.redistribute(source_mesh, [Replicate()]).to_local()
        expected_bf16 = full_logical_fp32.to(transfer_dtype)
        tgt_local = None
    else:
        torch.manual_seed(0)
        tgt_global = torch.zeros(tensor_shape, dtype=transfer_dtype, device=device)
        tgt_dt = distribute_tensor(tgt_global, target_mesh, list(target_placements))
        tgt_local = tgt_dt.to_local()
        expected_bf16 = torch.empty(tensor_shape, dtype=transfer_dtype, device=device)

    if is_source and rank == 0:
        for tr in range(source_ws, world_size):
            dist.send(expected_bf16, dst=tr)
    if not is_source:
        dist.recv(expected_bf16, src=0)

    m2m_map, src_slicers, tgt_slicers, partial_red = get_m2m_map(
        source_mesh=source_mesh,
        source_placements=list(source_placements),
        target_mesh=target_mesh,
        target_placements=list(target_placements),
        group=dist.group.WORLD,
        device=device,
    )

    source_ranks_list = list(range(source_ws))
    source_group = dist.new_group(ranks=source_ranks_list)
    source_partial_groups = _create_partial_groups(
        mesh_tensor=source_mesh.mesh,
        partial_reductions=partial_red,
        this_rank=rank,
        full_source_ranks=source_ranks_list,
        full_source_group=source_group,
    )

    chunks = map_to_chunk_ops(
        m2m_map=m2m_map,
        rank=rank,
        source_num_slicers=src_slicers,
        target_num_slicers=tgt_slicers,
        source_tensor=src_local if is_source else None,
        target_tensor=tgt_local,
        transfer_dtype=transfer_dtype,
        source_partial_groups=source_partial_groups or None,
    )

    match comm_method:
        case "chunk":
            chunk_comm(chunks=chunks)
        case "bucket":
            buckets = chunk_to_bucket_ops(chunks=chunks, bucket_size=256 * 1024 * 1024)
            bucket_comm(buckets=buckets)
        case _:
            raise ValueError(f"unknown comm_method: {comm_method}")

    if not is_source:
        got = tgt_dt.full_tensor()
        assert got.dtype == transfer_dtype, f"target dtype {got.dtype} != {transfer_dtype}"
        # bf16 has ~7-bit mantissa; rtol=1e-2 is tight enough that a
        # cast-first then bf16-reduce path on 1e3-scale inputs would fail.
        assert torch.allclose(got, expected_bf16, rtol=1e-2, atol=1.0), (
            f"mixed-dtype partial reshard mismatch on rank {rank}: "
            f"max diff {(got.float() - expected_bf16.float()).abs().max().item()}"
        )

    dist.destroy_process_group()


@pytest.mark.parametrize("comm_method", ["chunk", "bucket"])
def test_partial_mixed_dtype(comm_method):
    source_ws = 4
    target_ws = 4
    world_size = source_ws + target_ws
    tensor_shape = (8, 16)

    os.environ["MASTER_ADDR"] = "localhost"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])

    try:
        torch.multiprocessing.spawn(
            _run_partial_mixed_dtype,
            args=(world_size, source_ws, target_ws, tensor_shape, "cpu", comm_method),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        pytest.fail(f"mixed-dtype partial reshard failed: {e}")


@pytest.mark.parametrize(
    "source_mesh_shape, source_placements, target_mesh_shape, target_placements, tensor_shape, slice_view",
    [
        # 1. trainer MoE 3D → vLLM EP 1D, contiguous tensor.
        (
            (1, 2, 2),
            (Replicate(), Shard(0), Shard(0)),
            (4,),
            (Shard(0),),
            (8, 16, 32),
            False,
        ),
        # 2. trainer attn (2,2) → vLLM DPTP (2,2), identical layout.
        (
            (2, 2),
            (Replicate(), Shard(0)),
            (2, 2),
            (Replicate(), Shard(0)),
            (8, 16),
            False,
        ),
        # 3. trainer all-Replicate → vLLM (Replicate, Shard(0)).
        (
            (1, 4),
            (Replicate(), Replicate()),
            (2, 2),
            (Replicate(), Shard(0)),
            (8, 16),
            False,
        ),
        # 4. MoE 3D → EP 1D with non-contiguous slice-view (gate_up_proj pattern).
        (
            (1, 2, 2),
            (Replicate(), Shard(0), Shard(0)),
            (4,),
            (Shard(0),),
            (8, 16, 32),
            True,
        ),
        # 5. MoE 2D → EP 1D with slice-view.
        (
            (2, 2),
            (Shard(0), Shard(0)),
            (4,),
            (Shard(0),),
            (8, 16, 32),
            True,
        ),
        # 6. Source Partial (sum) on 1D mesh -> Replicate target (etha collapses
        # Partial -> Replicate via source-side all-reduce before send).
        (
            (4,),
            (Partial(),),
            (4,),
            (Replicate(),),
            (8, 16),
            False,
        ),
        # 7. Source Partial("avg") on 1D mesh -> Shard target.
        (
            (4,),
            (Partial("avg"),),
            (4,),
            (Shard(0),),
            (8, 16),
            False,
        ),
        # 8. 2D mesh with (Shard(0), Partial()) -> 1D Shard target.
        # The partial sub-group is per-row (smaller than full source).
        (
            (2, 2),
            (Shard(0), Partial()),
            (4,),
            (Shard(0),),
            (8, 16),
            False,
        ),
        # 9. 2D mesh with double Partial (full-source reduce) -> Replicate.
        (
            (2, 2),
            (Partial(), Partial()),
            (4,),
            (Replicate(),),
            (8, 16),
            False,
        ),
        # 10. Asymmetric 2D Partial-Partial with target smaller than source.
        # Forces the trace path to drop primaries for source ranks beyond
        # target_ws, so shadow expansion has to transitively reach every
        # Partial peer (e.g. rank 5 is only connected via shared sub-groups
        # with ranks 2,3,4; a non-transitive expand would leave it out and
        # deadlock the chunk-level reduce on sub-group [3,4,5]).
        (
            (2, 3),
            (Partial(), Partial()),
            (2,),
            (Replicate(),),
            (8, 16),
            False,
        ),
    ],
)
@pytest.mark.parametrize("comm_method", ["chunk", "bucket"])
def test_reshard(
    source_mesh_shape,
    source_placements,
    target_mesh_shape,
    target_placements,
    tensor_shape,
    slice_view,
    comm_method,
):
    source_ws = math.prod(source_mesh_shape)
    target_ws = math.prod(target_mesh_shape)
    world_size = source_ws + target_ws

    os.environ["MASTER_ADDR"] = "localhost"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])

    try:
        torch.multiprocessing.spawn(
            _run,
            args=(
                world_size,
                source_mesh_shape,
                source_placements,
                target_mesh_shape,
                target_placements,
                tensor_shape,
                "cpu",
                slice_view,
                comm_method,
            ),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        pytest.fail(f"reshard test failed: {e}")
