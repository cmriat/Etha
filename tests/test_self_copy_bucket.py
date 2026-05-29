"""A local (self-copy) chunk must write its target through both bucket paths.

A local chunk both ``is_source`` (reads a source slice) and ``is_target`` (writes
a target slice). ``Bucket.finalize`` writes when ``is_target``; both the
single-entry and bundled (multi-entry) bucket paths must land the copy.
"""

import os
import socket

import torch
import pytest
import torch.distributed as dist

from etha.comm import bucket_comm, chunk_to_bucket_ops
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


def _self_copy_chunk(tensor: torch.Tensor, src: int, dst: int) -> Chunk:
    """A local chunk that copies tensor[src:src+4] to tensor[dst:dst+4]."""
    return Chunk(
        chunk_shape=(4,),
        transport=Transport.LOCAL,
        is_source=True,
        is_target=True,
        src_rank=0,
        src_idx=(src // 4,),
        dst_ranks=(0,),
        dst_idx=(dst // 4,),
        src_slice=(slice(src, src + 4),),
        dst_slice=(slice(dst, dst + 4),),
        tensor=tensor,
    )


def test_self_copy_single_entry(single_rank_pg):
    t = torch.tensor([1.0, 2, 3, 4, 0, 0, 0, 0])
    bucket_comm(chunk_to_bucket_ops([_self_copy_chunk(t, src=0, dst=4)], bucket_size=1))
    assert torch.equal(t[4:8], torch.tensor([1.0, 2, 3, 4]))


def test_self_copy_bundled(single_rank_pg):
    t = torch.tensor([1.0, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0])
    chunks = [_self_copy_chunk(t, src=0, dst=8), _self_copy_chunk(t, src=4, dst=12)]
    buckets = chunk_to_bucket_ops(chunks, bucket_size=256 * 1024)
    assert len(buckets) == 1 and len(buckets[0].chunks) == 2  # multi-entry path
    bucket_comm(buckets)
    assert torch.equal(t[8:16], torch.tensor([1.0, 2, 3, 4, 5, 6, 7, 8]))
