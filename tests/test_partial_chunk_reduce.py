"""Correctness tests for chunk-level Partial -> Replicate reduce.

These tests exercise the streaming chunk-level reduce implementation in
isolation, comparing against PyTorch's ``DTensor.redistribute`` ground truth.
Etha's end-to-end Partial routing (``get_m2m_map`` + shadow expansion + chunk
pipeline) is covered separately in ``test_communication_replicate_shard``;
this file focuses on the reduce algorithm itself across corner cases:

* multi-dim meshes (1D / 2D / 3D)
* mixed ``Shard + Partial`` and double-``Partial`` placements
* all reduce ops (sum / avg / max / min)
* fp16 numerical fidelity with random data
* non-contiguous local tensors
* chunk size edge cases (> tensor, non-multiple of tensor)
"""

import os
import math
import socket
import logging

import torch
import pytest
import torch.distributed as dist
from torch.distributed._tensor import Shard, DTensor, Replicate, DeviceMesh
from torch.distributed.tensor.placement_types import Partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


REDUCE_OP_MAP = {
    "sum": dist.ReduceOp.SUM,
    "avg": dist.ReduceOp.AVG,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
}


def chunk_level_partial_to_replicate(
    local_tensor: torch.Tensor,
    mesh: DeviceMesh,
    placements: tuple,
    chunk_numel: int,
) -> torch.Tensor:
    """Reduce Partial-dim contributions via chunk-sized streaming all-reduce.

    For each ``Partial`` dim in ``placements``, run an all-reduce on the
    corresponding mesh sub-group, chunk by chunk. Returns the local tensor
    with all ``Partial`` dims collapsed to ``Replicate``.
    """
    contiguous_local = local_tensor.contiguous()
    flat_local = contiguous_local.view(-1)
    n = flat_local.numel()

    reductions = []
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Partial):
            op = REDUCE_OP_MAP[placement.reduce_op]
            group = mesh.get_group(mesh_dim)
            reductions.append((group, op))

    if not reductions:
        return contiguous_local.clone()

    out_flat = torch.empty_like(flat_local)
    chunk_buf = torch.empty(chunk_numel, dtype=local_tensor.dtype, device=local_tensor.device)

    for start in range(0, n, chunk_numel):
        end = min(start + chunk_numel, n)
        size = end - start
        view = chunk_buf[:size]
        view.copy_(flat_local[start:end])
        for group, op in reductions:
            dist.all_reduce(view, op=op, group=group)
        out_flat[start:end].copy_(view)

    return out_flat.view_as(contiguous_local)


def _compute_local_shape(tensor_shape, mesh_shape, placements):
    """Per-rank local shape after Shard partitioning (Partial / Replicate keep full shape)."""
    local = list(tensor_shape)
    for mesh_dim, p in enumerate(placements):
        if isinstance(p, Shard):
            assert local[p.dim] % mesh_shape[mesh_dim] == 0, (
                f"tensor dim {p.dim}={local[p.dim]} not divisible by mesh dim {mesh_dim}={mesh_shape[mesh_dim]}"
            )
            local[p.dim] //= mesh_shape[mesh_dim]
    return tuple(local)


def _run(
    rank: int,
    world_size: int,
    mesh_shape: tuple[int, ...],
    placements: tuple,
    tensor_shape: tuple[int, ...],
    chunk_numel: int,
    dtype_str: str,
    non_contiguous: bool,
):
    """Build a Partial DTensor, run chunk-level reduce, compare against PyTorch."""
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    mesh = DeviceMesh("cpu", torch.arange(world_size).view(mesh_shape))
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]

    local_shape = _compute_local_shape(tensor_shape, mesh_shape, placements)

    torch.manual_seed(1000 + rank)
    if non_contiguous:
        wide_shape = list(local_shape)
        wide_shape[-1] *= 2
        wide = torch.randn(wide_shape, dtype=dtype)
        local = wide[..., : local_shape[-1] * 2 : 2]
        assert not local.is_contiguous(), "non-contiguous setup failed"
    else:
        local = torch.randn(local_shape, dtype=dtype)

    dt = DTensor.from_local(local, mesh, list(placements))
    target_placements = [Replicate() if isinstance(p, Partial) else p for p in placements]
    truth = dt.redistribute(mesh, target_placements).to_local()

    result = chunk_level_partial_to_replicate(local, mesh, placements, chunk_numel)

    # Low-precision dtypes accumulate differently under chunked vs full reduce
    # because the backend picks reduction order per tensor size. Tolerances
    # are set to the natural ULP scale near typical sum magnitudes (bf16 has
    # only 7 mantissa bits, so ULP near sums of N(0,1) values is ~3e-2).
    if dtype == torch.float16:
        rtol, atol = 5e-3, 5e-3
    elif dtype == torch.bfloat16:
        rtol, atol = 5e-2, 5e-2
    else:
        rtol, atol = 1e-5, 1e-5

    max_diff = (result.float() - truth.float()).abs().max().item()
    assert torch.allclose(result, truth, rtol=rtol, atol=atol), (
        f"rank {rank}: chunk-level diverges from DTensor.redistribute; max diff = {max_diff} "
        f"(rtol={rtol}, atol={atol}, dtype={dtype_str}, placements={placements}, chunk={chunk_numel})"
    )

    dist.destroy_process_group()


_CASES = [
    # === 1D mesh, single Partial, each reduce op ===
    ((4,), (Partial("sum"),), (8, 16), 32, "fp32", False),
    ((4,), (Partial("avg"),), (8, 16), 32, "fp32", False),
    ((4,), (Partial("max"),), (8, 16), 32, "fp32", False),
    ((4,), (Partial("min"),), (8, 16), 32, "fp32", False),
    # === 2D mesh, Partial combined with other placements ===
    ((2, 2), (Replicate(), Partial()), (8, 16), 32, "fp32", False),
    ((2, 2), (Partial(), Replicate()), (8, 16), 32, "fp32", False),
    # Shard + Partial -- the key correctness corner case
    ((2, 2), (Shard(0), Partial()), (8, 16), 32, "fp32", False),
    ((2, 2), (Partial(), Shard(0)), (8, 16), 32, "fp32", False),
    ((2, 2), (Shard(1), Partial()), (8, 16), 32, "fp32", False),
    # Double Partial -- reduce on both mesh dims
    ((2, 2), (Partial(), Partial()), (8, 16), 32, "fp32", False),
    ((2, 2), (Partial("avg"), Partial("avg")), (8, 16), 32, "fp32", False),
    # === 3D mesh, mixed (mirrors the trainer MoE shape used elsewhere) ===
    ((1, 2, 2), (Replicate(), Partial(), Shard(0)), (8, 16), 32, "fp32", False),
    ((2, 2, 2), (Shard(0), Partial(), Replicate()), (16, 32), 64, "fp32", False),
    # === Dtype + numerical fidelity (random data forces accumulation error) ===
    ((4,), (Partial(),), (64, 64), 64, "fp16", False),
    ((4,), (Partial(),), (64, 64), 64, "bf16", False),
    ((4,), (Partial("avg"),), (64, 64), 64, "fp16", False),
    # === Non-contiguous local tensor ===
    ((4,), (Partial(),), (8, 16), 32, "fp32", True),
    ((2, 2), (Shard(0), Partial()), (8, 16), 32, "fp32", True),
    # === Chunk size edge cases ===
    ((4,), (Partial(),), (8, 16), 1024, "fp32", False),  # chunk > tensor
    ((4,), (Partial(),), (10, 13), 7, "fp32", False),  # non-multiple
    ((4,), (Partial(),), (3, 5), 1, "fp32", False),  # chunk = 1
]


@pytest.mark.parametrize(
    "mesh_shape, placements, tensor_shape, chunk_numel, dtype_str, non_contiguous",
    _CASES,
)
def test_partial_chunk_reduce(mesh_shape, placements, tensor_shape, chunk_numel, dtype_str, non_contiguous):
    """Chunk-level reduce must agree with DTensor.redistribute Partial -> Replicate."""
    world_size = math.prod(mesh_shape)

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
                mesh_shape,
                placements,
                tensor_shape,
                chunk_numel,
                dtype_str,
                non_contiguous,
            ),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        pytest.fail(f"chunk-level reduce test failed: {e}")
