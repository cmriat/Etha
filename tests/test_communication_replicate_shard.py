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
from torch.distributed._tensor import Shard, Replicate, DeviceMesh, distribute_tensor

from etha.comm import chunk_comm, get_m2m_map, map_to_chunk_ops

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    if slice_view:
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

    m2m_map, src_slicers, tgt_slicers = get_m2m_map(
        source_mesh=source_mesh,
        source_placements=list(source_placements),
        target_mesh=target_mesh,
        target_placements=list(target_placements),
        group=dist.group.WORLD,
        device=device,
    )

    chunks = map_to_chunk_ops(
        m2m_map=m2m_map,
        rank=rank,
        source_num_slicers=src_slicers,
        target_num_slicers=tgt_slicers,
        source_tensor=src_local,
        target_tensor=tgt_local,
    )
    chunk_comm(chunks=chunks)

    if not is_source:
        if slice_view:
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
    ],
)
def test_reshard(
    source_mesh_shape,
    source_placements,
    target_mesh_shape,
    target_placements,
    tensor_shape,
    slice_view,
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
            ),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        pytest.fail(f"reshard test failed: {e}")
