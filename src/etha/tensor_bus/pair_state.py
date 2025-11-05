"""State."""

import torch
import msgspec
import torch.distributed as dist

from etha.comm.ir import SourceChunk, TargetChunk


class M2MMap(msgspec.Struct):
    """Mesh to mesh topology using M2MMap (shape-independent).

    M2MMap structure: dict[src_rank, dict[src_idx, list[tuple[dst_rank, dst_idx]]]]
    """

    m2m_map: dict[int, dict[tuple, list[tuple[int, tuple]]]] | None = None
    source_num_slicers: list[int] | None = None  # How source tensor is partitioned
    target_num_slicers: list[int] | None = None  # How target tensor is partitioned


class PairState(msgspec.Struct):
    """State of a registered Pair (stored in Daemon)."""

    pair_name: str
    local_name: str  # Local peer name (e.g., "inference", "training")
    local_ranks: list[int]  # Ranks for local peer (e.g., [0, 1, 2, ..., 7])
    remote_name: str  # Remote peer name
    remote_ranks: list[int]  # Ranks for remote peer (e.g., [8, 9, ..., 23])
    pair_size: int  # Total number of ranks in the pair
    local_group: dist.ProcessGroup  # Local process group
    pair_group: dist.ProcessGroup  # Pair process group
    status: str  # "matched"

    # Topology layer: M2M maps (shape-independent, reusable)
    m2m_send: M2MMap | None = None  # Map for sending (local -> remote)
    m2m_recv: M2MMap | None = None  # Map for receiving (remote -> local)

    # Data layer: Per-tensor storage
    tensors: dict[str, torch.Tensor] = {}  # tensor_name -> tensor mapping

    # Execution layer: Unified chunk lists (one pair per direction)
    send_chunks: list[SourceChunk | TargetChunk] | None = None  # Unified send chunks
    recv_chunks: list[SourceChunk | TargetChunk] | None = None  # Unified recv chunks
