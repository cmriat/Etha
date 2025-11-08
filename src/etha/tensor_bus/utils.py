"""Utility functions for TensorBus."""

import os
import logging

import torch.multiprocessing.reductions as mp_reductions

logger = logging.getLogger(__name__)

# CUDA device mapping patch for Agent-side deserialization
_original_rebuild_cuda_tensor = None


def _patched_rebuild_cuda_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    """Patched rebuild_cuda_tensor that uses agent's LOCAL_RANK.

    When client serializes tensor with device 'cuda:X', agent needs to
    deserialize it as 'cuda:{LOCAL_RANK}' because data is already
    on the correct GPU via CUDA IPC.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    original_device = storage_device
    storage_device = local_rank  # Must be int, not "cuda:X" string
    logger.debug(
        f"[Agent] CUDA device mapping: cuda:{original_device} -> cuda:{storage_device} (LOCAL_RANK={local_rank})"
    )

    return _original_rebuild_cuda_tensor(
        tensor_cls,
        tensor_size,
        tensor_stride,
        tensor_offset,
        storage_cls,
        dtype,
        storage_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        requires_grad,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    )


def setup_cuda_rebuild_patch():
    """Setup global monkey patch for rebuild_cuda_tensor."""
    global _original_rebuild_cuda_tensor
    if _original_rebuild_cuda_tensor is None:
        _original_rebuild_cuda_tensor = mp_reductions.rebuild_cuda_tensor
        mp_reductions.rebuild_cuda_tensor = _patched_rebuild_cuda_tensor
        logger.info("[Agent] CUDA rebuild_cuda_tensor patch installed")
