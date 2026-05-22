"""State."""

import msgspec
import torch.distributed as dist


class M2MMap(msgspec.Struct):
    """Mesh to mesh topology using M2MMap (shape-independent).

    M2MMap structure: dict[src_rank, dict[src_idx, list[tuple[dst_rank, dst_idx]]]]
    """

    m2m_map: dict[int, dict[tuple, list[tuple[int, tuple]]]] | None = None
    source_num_slicers: list[int] | None = None  # How source tensor is partitioned
    target_num_slicers: list[int] | None = None  # How target tensor is partitioned
    # Partial placements found on the source mesh, as (mesh_dim_idx, reduce_op).
    # The caller must all-reduce each Partial dim on the corresponding source
    # sub-group before send; empty when source has no Partial.
    source_partial_reductions: list[tuple[int, str]] = []


class PairState(msgspec.Struct):
    """State of a registered Pair.

    PairState is created once per pair via register_pair() and represents
    the communication topology. It contains NO tensor data or execution state.
    All tensor data and execution plans are now stored in BatchState.
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
