"""State."""

import torch
import msgspec
import torch.distributed as dist

from etha.comm.chunk_ops import SourceChunk, TargetChunk


class M2MMap(msgspec.Struct):
    """Mesh to mesh topology map (shape-independent)."""

    forward_map: dict  # src_rank -> {src_idx: [(dst_rank, dst_idx), ...]}
    reverse_map: dict  # dst_rank -> {dst_idx: [(src_rank, src_idx), ...]}
    source_num_slicers: list[int]  # Number of slices per dimension for source
    target_num_slicers: list[int]  # Number of slices per dimension for target


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
    m2m_map_send: M2MMap | None = None  # Map for sending (local -> remote)
    m2m_map_recv: M2MMap | None = None  # Map for receiving (remote -> local)

    # Data layer: Per-tensor storage
    tensors: dict[str, torch.Tensor] = {}  # tensor_name -> tensor mapping
    # Per-tensor IR: tensor_name -> (send_ir, recv_ir)
    # Each IR is a tuple of (source_chunks, target_chunks)
    tensor_irs: dict[
        str, tuple[tuple[list[SourceChunk], list[TargetChunk]], tuple[list[SourceChunk], list[TargetChunk]]]
    ] = {}
