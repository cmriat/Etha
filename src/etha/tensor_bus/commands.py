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
    """Transfer tensor command."""

    pair_name: str
    transfer_type: Literal["send", "recv"]


class RegisterTensorBatch(BaseCommand):
    """Register multiple tensors for zero-copy sharing between processes.

    Batch Register Tensors for improved efficiency when registering
    multiple tensors. Reduces LMDB cross-process communication overhead by
    sending all tensors in a single command instead of multiple individual commands.
    """

    pair_name: str
    tensor_names: list[str]
    tensor_payloads: list[memoryview]


class RegisterPair(BaseCommand):
    """Register to a Pair (peer-to-peer communication endpoint).

    Design:
    - Pair is symmetric: both peers can send() or recv()
    - Similar to RDMA Queue Pair (QP)
    - Registration writes to TCPStore and polls until both peers are ready
    - Includes local device mesh and placement information for optimized tensor transfer

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
    """Query status for a pair."""

    pair_name: str
    state_name: str


Message = Transfer | RegisterTensorBatch | RegisterPair | QueryStatus
