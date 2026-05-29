"""State."""

from dataclasses import dataclass

import torch.distributed as dist

from etha.comm.ir import M2MMap


@dataclass(kw_only=True)
class PairState:
    """Runtime state of a registered Pair, held in-memory on the Agent.

    Created once per pair via register_pair(). Holds live process-group handles
    and the shape-independent M2M topology; never crosses a process boundary
    (the Agent only exposes scalar signals to the Client over LMDB).
    """

    pair_name: str
    local_name: str  # Local peer name (e.g., "inference", "training")
    local_ranks: list[int]  # Ranks for local peer (e.g., [0, 1, 2, ..., 7])
    remote_name: str  # Remote peer name
    remote_ranks: list[int]  # Ranks for remote peer (e.g., [8, 9, ..., 23])
    pair_size: int  # Total number of ranks in the pair
    local_group: dist.ProcessGroup  # Local process group
    pair_group: dist.ProcessGroup  # Pair process group
    local_is_first: bool  # Whether local is first in the pair

    # Topology layer: M2M maps (shape-independent, reusable across batches)
    m2m_send: M2MMap | None = None  # Map for sending (local -> remote)
    m2m_recv: M2MMap | None = None  # Map for receiving (remote -> local)

    # NCCL sub-groups for the partial dims on the *local* side of m2m_send
    # (None if no Partial placements). Each entry: (sub_group, reduce_op_str);
    # populated only on the sender side of the pair. The send pipeline runs
    # an in-place all-reduce on these groups before the actual P2P send.
    source_partial_groups: list[tuple[dist.ProcessGroup, str]] | None = None
