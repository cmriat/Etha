"""Get chunks from m2m map."""

import torch

from .ir import (
    BaseChunk,
    RecvChunk,
    SendChunk,
    TransferType,
    SelfCopyChunk,
)
from .utils import (
    get_slicer_tuples,
    get_slice_from_multi_index,
    get_or_create_process_group,
)


def calculate_chunk_shape(
    num_slicers: list[int],
    tensor_shape: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if tensor_shape is None:
        return ()
    chunk_shape = tuple(tensor_shape[dim] // num_slicers[dim] for dim in range(len(tensor_shape)))
    return chunk_shape


def map_to_chunk_ops(
    m2m_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    rank: int,
    source_num_slicers: list[int],
    target_num_slicers: list[int],
    source_tensor: torch.Tensor | None = None,
    target_tensor: torch.Tensor | None = None,
    target_dtype: torch.dtype | None = None,
) -> list[BaseChunk]:
    if not m2m_map:
        return []
    source_tensor_shape = source_tensor.shape if source_tensor is not None else None
    target_tensor_shape = target_tensor.shape if target_tensor is not None else None
    source_slicer_tuples = None
    source_num_slicers_extended = None
    if source_tensor_shape is not None:
        source_num_slicers_extended = (source_num_slicers + [1] * len(source_tensor_shape))[: len(source_tensor_shape)]
        source_slicer_tuples = get_slicer_tuples(source_tensor_shape, source_num_slicers_extended)
    target_slicer_tuples = None
    target_num_slicers_extended = None
    if target_tensor_shape is not None:
        target_num_slicers_extended = (target_num_slicers + [1] * len(target_tensor_shape))[: len(target_tensor_shape)]
        target_slicer_tuples = get_slicer_tuples(target_tensor_shape, target_num_slicers_extended)
    broadcast_groups = set()
    for src_rank, src_map in m2m_map.items():
        for _, dst_list in src_map.items():
            if len(dst_list) > 1:
                group_ranks = (src_rank,) + tuple(sorted({r for r, _ in dst_list}))
                broadcast_groups.add(group_ranks)
    for group_ranks in sorted(broadcast_groups):
        get_or_create_process_group(list(group_ranks))
    chunks: list[BaseChunk] = []
    for src_rank, src_map in m2m_map.items():
        for src_idx, dst_list in src_map.items():
            if len(dst_list) > 1:
                transfer_type = TransferType.BROADCAST
            else:
                transfer_type = TransferType.P2P
            dst_ranks: tuple[int, ...] = tuple(sorted({r for r, _ in dst_list}))
            if src_rank == rank:
                src_slice_tuples: tuple[slice, ...] = ()
                if source_slicer_tuples is not None and source_num_slicers_extended is not None:
                    src_slice_tuples = get_slice_from_multi_index(
                        src_idx, source_num_slicers_extended, source_slicer_tuples
                    )

                source_chunk = SendChunk(
                    chunk_shape=calculate_chunk_shape(source_num_slicers_extended, source_tensor_shape),
                    transfer_type=transfer_type,
                    is_source=True,
                    src_rank=rank,
                    src_idx=src_idx,
                    dst_ranks=dst_ranks,
                    slice_tuples=src_slice_tuples,
                    tensor=source_tensor,
                    target_dtype=target_dtype,
                )
                chunks.append(source_chunk)
            for dst_rank, dst_idx in dst_list:
                if dst_rank == rank:
                    if src_rank == rank:
                        actual_transfer_type = TransferType.SELF_COPY
                    else:
                        actual_transfer_type = transfer_type
                    dst_slice_tuples: tuple[slice, ...] = ()
                    if target_slicer_tuples is not None and target_num_slicers_extended is not None:
                        dst_slice_tuples = get_slice_from_multi_index(
                            dst_idx, target_num_slicers_extended, target_slicer_tuples
                        )
                    dst_src_slice_tuples: tuple[slice, ...] = ()
                    if (
                        actual_transfer_type == TransferType.SELF_COPY
                        and source_slicer_tuples is not None
                        and source_num_slicers_extended is not None
                    ):
                        dst_src_slice_tuples = get_slice_from_multi_index(
                            src_idx, source_num_slicers_extended, source_slicer_tuples
                        )

                    if actual_transfer_type == TransferType.SELF_COPY:
                        target_chunk = SelfCopyChunk(
                            chunk_shape=calculate_chunk_shape(target_num_slicers_extended, target_tensor_shape),
                            transfer_type=actual_transfer_type,
                            is_source=True,
                            src_rank=src_rank,
                            src_idx=src_idx,
                            dst_ranks=dst_ranks,
                            dst_idx=dst_idx,
                            slice_tuples=dst_slice_tuples,
                            src_slice_tuples=src_slice_tuples,
                            tensor=target_tensor,
                        )
                    else:
                        target_chunk = RecvChunk(
                            chunk_shape=calculate_chunk_shape(target_num_slicers_extended, target_tensor_shape),
                            transfer_type=actual_transfer_type,
                            is_source=False,
                            src_rank=src_rank,
                            src_idx=src_idx,
                            dst_ranks=dst_ranks,
                            dst_idx=dst_idx,
                            slice_tuples=dst_slice_tuples,
                            tensor=target_tensor,
                            target_dtype=target_dtype,
                        )
                    chunks.append(target_chunk)
    return chunks
