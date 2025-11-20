"""Command (Host to Tensor Bus) Definitions and State Structures."""

from typing import Literal

import msgspec


class BaseCommand(msgspec.Struct, tag=True, kw_only=True):
    """Base class for all Tensor Bus commands.

    Features:
    - Auto-tagging: Uses class name as type tag
    - Common timestamp field for all commands
    - Optional semaphore for completion notification
    """

    timestamp: float | None = None
    semaphore_name: str | None = None


class Transfer(BaseCommand):
    """Transfer tensor command for a specific batch."""

    batch_id: str
    transfer_type: Literal["send", "recv"]


class RegisterTensors(BaseCommand):
    """Register multiple tensors for zero-copy sharing between processes.

    Creates a new batch with a unique batch_id. Multiple tensors can be
    registered across different pairs in a single batch, enabling efficient
    cross-pair execution via flattened chunks/buckets.
    """

    batch_id: str
    tensors: list[tuple[str, memoryview]]  # (pair_name, tensor_payload)
    bucket_size: int | None = None  # Optional bucket size in bytes


class InitPair(BaseCommand):
    """Init a Device Mesh + Placement to Device Mesh + Placement pair.

    Args:
        pair_name: Unique identifier for this pair (e.g., "obs", "action")
        local_name: Name of local peer (e.g., "inference", "training")
        expected_world_size: Number of ranks for local peer
        remote_name: Name of remote peer (explicit pairing)
        mesh_shape_payload: Serialized mesh shape tuple as memoryview
        placements_payload: Serialized placements tuple as memoryview
    """

    pair_name: str
    local_name: str
    expected_world_size: int
    remote_name: str
    mesh_shape_payload: memoryview | None = None
    placements_payload: memoryview | None = None


class QueryStatus(BaseCommand):
    """Query status for a batch."""

    batch_id: str
    state_name: str


class CleanupBatch(BaseCommand):
    """Cleanup a batch's state in the agent."""

    batch_id: str


Message = Transfer | RegisterTensors | InitPair | QueryStatus | CleanupBatch
