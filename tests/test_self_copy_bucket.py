"""A local (self-copy) chunk must write its target through both paths.

A local chunk both ``is_source`` (reads a source slice) and ``is_target`` (writes
a target slice). ``Bucket.finalize`` writes when ``is_target``; both the single
and bundled bucket paths must land the copy in the target tensor — same as the
direct chunk path.
"""

import os
import socket

import torch
import pytest
import torch.distributed as dist

from etha.comm import chunk_comm, bucket_comm, chunk_to_bucket_ops
from etha.comm.ir import Chunk, Transport


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def single_rank_pg():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(_free_port())
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _make_self_copy_chunk(tensor: torch.Tensor) -> Chunk:
    """Mirror map_to_chunk_ops' local construction.

    Reads the src slice, writes the dst slice, single tensor, is_source + is_target.
    """
    return Chunk(
        chunk_shape=(4,),
        transport=Transport.LOCAL,
        is_source=True,
        is_target=True,
        src_rank=0,
        src_idx=(0,),
        dst_ranks=(0,),
        dst_idx=(1,),
        src_slice=(slice(0, 4),),  # src
        dst_slice=(slice(4, 8),),  # dst
        tensor=tensor,
    )


def test_self_copy_chunk_path(single_rank_pg):
    t = torch.tensor([1.0, 2, 3, 4, 0, 0, 0, 0])
    chunk_comm([_make_self_copy_chunk(t)])
    assert torch.equal(t[4:8], torch.tensor([1.0, 2, 3, 4]))


def test_self_copy_bucket_path(single_rank_pg):
    t = torch.tensor([1.0, 2, 3, 4, 0, 0, 0, 0])
    buckets = chunk_to_bucket_ops([_make_self_copy_chunk(t)], bucket_size=256 * 1024)
    bucket_comm(buckets)
    assert torch.equal(t[4:8], torch.tensor([1.0, 2, 3, 4]))
