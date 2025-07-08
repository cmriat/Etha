"""P2P Communication for Distributed Tensor Operations."""

from typing import Dict, List, Optional
import itertools

import torch
import torch.distributed as dist

def p2p_communicate(
    rank: int,
    send_map: Dict[int, List[int]],
    reverse_map: Dict[int, List[List[int]]],
    local_tensor_shard: torch.Tensor,
    local_tensor_shape: torch.Size,
    target_shape: torch.Size
) -> Optional[torch.Tensor]:
    """
    Performs P2P communication to redistribute tensor shards across devices.
    This version uses non-blocking sends/receives and performs a nested
    concatenation based on the reverse_map.

    Args:
        rank (int): The rank of the current process.
        send_map (Dict[int, List[int]]): A map from a source rank to a list of target ranks.
        reverse_map (Dict[int, List[List[int]]]): A map from a target rank to a list of lists of source ranks,
                                                   defining the order of concatenation.
        local_tensor_shard (torch.Tensor): The local tensor shard on the current device.
        target_shard_tensor_shape (torch.Size): The shape of the target tensor shard.

    Returns:
        Optional[torch.Tensor]: The fully assembled tensor on receiver ranks, otherwise None.
    """
    received_tensors = {}

    # Non-blocking sends
    send_reqs = []
    if rank in send_map:
        local_shape = local_tensor_shape

        # 如果本地形状和目标形状相同，则向每个目标发送相同的张量
        if local_shape == target_shape:
            for target_rank in send_map[rank]:
                if target_rank == rank:
                    # 如果目标是自己，则不需要发送
                    received_tensors[target_rank] = local_tensor_shard
                else:
                    req = dist.isend(tensor=local_tensor_shard, dst=target_rank)
                    send_reqs.append(req)
        else:
            # 否则，为每个维度生成切片规则
            slicers_per_dim = []
            for l, t in zip(local_shape, target_shape):
                if l > t:
                    # 如果本地维度大于目标维度，则切分
                    num_chunks = l // t
                    slicers_per_dim.append([slice(i * t, (i + 1) * t) for i in range(num_chunks)])
                else:
                    # 否则，取整个维度
                    slicers_per_dim.append([slice(None)])

            # 使用 itertools.product 组合所有维度的切片
            all_slicer_tuples = list(itertools.product(*slicers_per_dim))
            num_slicers = len(all_slicer_tuples)
            num_targets = len(send_map[rank])

            # 将切片循环发送到目标 rank
            for i, target_rank in enumerate(send_map[rank]):
                slicer_index = i % num_slicers
                slicer_tuple = all_slicer_tuples[slicer_index]
                chunk_to_send = local_tensor_shard[slicer_tuple]
                if target_rank == rank:
                    # 如果目标是自己，则不需要发送
                    received_tensors[target_rank] = chunk_to_send
                else:
                    req = dist.isend(tensor=chunk_to_send.contiguous(), dst=target_rank)
                    send_reqs.append(req) 

    # Non-blocking receives
    recv_reqs = []
    if rank in reverse_map:
        # --- Calculate the shape of incoming chunks ---
        # This assumes all source ranks have the same local_tensor_shard shape
        # as the current rank. The calculation must mirror the sender's slicing logic.
        source_shape_assumption = local_tensor_shape
        chunk_dims = []
        for s_dim, t_dim in zip(source_shape_assumption, target_shape):
            if s_dim > t_dim:
                # If the source dimension is sliced, the chunk's dimension will be the target's.
                chunk_dims.append(t_dim)
            else:
                # Otherwise, the chunk's dimension is the same as the source's.
                chunk_dims.append(s_dim)
        incoming_chunk_shape = torch.Size(chunk_dims)
        # --- End of shape calculation ---

        all_source_ranks = [
            source_rank
            for sources_for_slice in reverse_map[rank]
            for source_rank in sources_for_slice
        ]
        for source_rank in all_source_ranks:
            if source_rank != rank:
                # Use the calculated shape for the receive buffer.
                recv_buffer = torch.empty(incoming_chunk_shape)
                req = dist.irecv(tensor=recv_buffer, src=source_rank)
                recv_reqs.append(req)
                received_tensors[source_rank] = recv_buffer

    # Wait for all operations to complete
    for req in send_reqs:
        req.wait()
    for req in recv_reqs:
        req.wait()

    # Concatenate tensors on receiver ranks based on the reverse_map structure
    if rank in reverse_map:
        
        outer_slices = []
        for sources_for_slice in reverse_map[rank]:
            if not sources_for_slice:
                continue
            
            inner_tensors_to_cat = [received_tensors[source_rank] for source_rank in sources_for_slice]
            # Perform inner concatenation
            inner_slice = torch.cat(inner_tensors_to_cat, dim=1)
            outer_slices.append(inner_slice)

        if not outer_slices:
            return None

        # Perform outer concatenation
        final_tensor = torch.cat(outer_slices, dim=0)
        return final_tensor

    # Ranks that are not receivers return None
    return None
