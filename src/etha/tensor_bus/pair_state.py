"""State."""

import torch
import msgspec
import torch.distributed as dist


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
    tensors: dict[str, torch.Tensor] = {}  # tensor_name -> tensor mapping
    p2p_map_send: dict | None = None  # P2P transfer map for sending optimized communication
    p2p_map_recv: dict | None = None  # P2P transfer map for receiving optimized communication
