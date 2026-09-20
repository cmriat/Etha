"""Command (Host to Tensor Bus) Definitions and State Structures."""

from typing import Literal

import msgspec


def command_error_key(semaphore_name: str) -> bytes:
    """LMDB key used to report a command failure to its waiting client."""
    return f"command:{semaphore_name}:error".encode()


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
    """Transfer tensor command for a specific batch.

    On a dual-endpoint (colocated) batch the agent hosts both sides of the
    pair, so the command alone cannot tell which side issued it: ``role`` names
    the issuing client's side (its ``init_pair`` ``local_name``). ``send`` from
    role R executes the R→other direction; ``recv`` from role R executes the
    other→R direction. Split (single-role) agents leave it None.

    ``sync_round`` identifies the weight-sync round within one batch: a
    direction-round executes exactly once, only after BOTH roles' commands
    for that round arrived (source-ready and dest-ready), and re-issues of an
    already-executed or older round are acknowledged without re-execution.
    Monotonic per (batch, direction); round 0 is the first sync. Split
    batches ignore it.
    """

    batch_id: str
    transfer_type: Literal["send", "recv"]
    role: str | None = None
    sync_round: int = 0


class RegisterTensors(BaseCommand):
    """Register multiple tensors for zero-copy sharing between processes.

    Creates a new batch with a unique batch_id. Multiple tensors can be
    registered across different pairs in a single batch, enabling efficient
    cross-pair execution via flattened chunks/buckets.

    ``role`` is required when a pair of the batch is dual-endpoint (colocated):
    both sides of the pair register into the same batch on the same agent, and
    the role names which side this registration's tensors belong to. It must
    match one of the pair's peer names. Split pairs leave it None.
    """

    batch_id: str
    tensors: list[tuple[str, memoryview]]  # (pair_name, tensor_payload)
    bucket_size: int | None = None  # Optional bucket size in bytes
    role: str | None = None


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
