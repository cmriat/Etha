"""Communication Lowering - Convert topology maps to IR chunks."""

from collections import defaultdict

from etha.comm.utils import get_or_create_process_group

from .chunk_ir import SourceChunk, TargetChunk


def build_broadcast_plan(
    forward_map: dict[int, dict[tuple, list[int]]],
) -> tuple[dict[tuple[int, tuple[int, ...]], list[tuple]], set[tuple[int, tuple]]]:
    """Build broadcast plan for 1-to-many transfers.

    Returns:
        broadcast_plan: Maps (src, tuple(targets)) -> [source_idx...]
        broadcast_keys: Set of (src, source_idx) that use broadcast
    """
    broadcast_plan: dict[tuple[int, tuple[int, ...]], list[tuple]] = defaultdict(list)
    for src in sorted(forward_map.keys()):
        inner = forward_map[src]
        for source_idx in sorted(inner.keys()):
            targets = inner[source_idx]
            other_targets = sorted([r[0] for r in targets if r[0] != src])
            if len(other_targets) > 1:
                broadcast_plan[(src, tuple(other_targets))].append(source_idx)

    broadcast_keys: set[tuple[int, tuple]] = set()
    for (src, _targets), idx_list in broadcast_plan.items():
        for sx in idx_list:
            broadcast_keys.add((src, sx))

    # Create all subgroups in a consistent order across all ranks
    for group_key in sorted(broadcast_plan.keys(), key=lambda k: (k[0], k[1])):
        group_ranks = [group_key[0]] + list(group_key[1])
        get_or_create_process_group(group_ranks)

    return broadcast_plan, broadcast_keys


def map_to_ops(
    forward_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    reverse_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    source_num_slicers: list[int],
    target_num_slicers: list[int],
    target_tensor_shape: tuple[int, ...],
    rank: int,
) -> tuple[list[SourceChunk], list[TargetChunk]]:
    """Lower topology maps to chunk-level IR.

    Pure planning function - no execution, no state.

    Args:
        forward_map: src_rank -> {src_idx: [(dst_rank, dst_idx), ...]}
        reverse_map: dst_rank -> {dst_idx: [(src_rank, src_idx), ...]}
        source_num_slicers: Partitioning of source tensor
        target_num_slicers: Partitioning of target tensor
        target_tensor_shape: Shape of final target tensor
        rank: Current process rank

    Returns:
        (source_chunks, target_chunks):
            - source_chunks: Chunks this rank needs to SEND
            - target_chunks: Chunks this rank needs to RECEIVE
    """
    # Build broadcast plan to identify 1-to-N transfers
    broadcast_plan, broadcast_keys = build_broadcast_plan(forward_map)

    source_chunks: list[SourceChunk] = []
    target_chunks: list[TargetChunk] = []

    chunk_id = 0

    # === Generate SourceChunks from forward_map ===
    if rank in forward_map:
        for source_idx in sorted(forward_map[rank].keys()):
            targets = forward_map[rank][source_idx]  # [(dst_rank, dst_idx), ...]

            # Extract dst_ranks (excluding self)
            dst_ranks = sorted({r[0] for r in targets if r[0] != rank})

            if not dst_ranks:
                continue

            # Determine transfer type
            if (rank, source_idx) in broadcast_keys:
                transfer_type = "broadcast"
                group_key = (rank, tuple(dst_ranks))
            else:
                transfer_type = "p2p"
                group_key = None

            # Calculate chunk shape (will be set properly during preparation)
            # For source chunks, we don't have tensor_shape yet, will be updated during prepare
            chunk_shape = _calculate_chunk_shape(source_num_slicers, None)

            source_chunk = SourceChunk(
                chunk_id=chunk_id,
                chunk_shape=chunk_shape,
                transfer_type=transfer_type,
                src_rank=rank,
                src_idx=source_idx,
                dst_ranks=dst_ranks,
                group_key=group_key,
            )
            # Note: buffer will be set during prepare_send_buffers()
            source_chunks.append(source_chunk)
            chunk_id += 1

    # === Generate TargetChunks from reverse_map ===
    if rank in reverse_map:
        for target_idx in sorted(reverse_map[rank].keys()):
            src_list = reverse_map[rank][target_idx]  # [(src_rank, src_idx), ...]

            # Usually len(src_list) == 1 (one slot receives from one source)
            # But handle multiple sources just in case
            for src_rank, src_idx in src_list:
                # Determine transfer type
                if src_rank == rank:
                    transfer_type = "self_copy"
                    group_key = None
                elif (src_rank, src_idx) in broadcast_keys:
                    transfer_type = "broadcast"
                    # Find the dst_ranks for this broadcast
                    # From broadcast_plan: (src_rank, tuple(dst_ranks)) -> [src_idx, ...]
                    dst_ranks = None
                    for (bs, bdst), idx_list in broadcast_plan.items():
                        if bs == src_rank and src_idx in idx_list:
                            dst_ranks = bdst
                            break
                    group_key = (src_rank, dst_ranks) if dst_ranks else None
                else:
                    transfer_type = "p2p"
                    group_key = None

                # Calculate chunk shape
                chunk_shape = _calculate_chunk_shape(target_num_slicers, target_tensor_shape)

                target_chunk = TargetChunk(
                    chunk_id=chunk_id,
                    chunk_shape=chunk_shape,
                    transfer_type=transfer_type,
                    dst_rank=rank,
                    dst_idx=target_idx,
                    src_rank=src_rank,
                    src_idx=src_idx,
                    group_key=group_key,
                )
                # Note: buffer will be set during prepare_recv_buffers()
                target_chunks.append(target_chunk)
                chunk_id += 1

    return source_chunks, target_chunks


def _calculate_chunk_shape(
    num_slicers: list[int],
    tensor_shape: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Calculate chunk shape from num_slicers and tensor shape.

    Args:
        num_slicers: Number of slices per dimension
        tensor_shape: Full tensor shape (if known)

    Returns:
        Shape of the chunk, or empty tuple if tensor_shape is None
    """
    if tensor_shape is None:
        return ()

    # Extend num_slicers to match tensor dimensions
    num_slicers = num_slicers + [1] * (len(tensor_shape) - len(num_slicers))

    chunk_shape = tuple(tensor_shape[dim] // num_slicers[dim] for dim in range(len(tensor_shape)))

    return chunk_shape
