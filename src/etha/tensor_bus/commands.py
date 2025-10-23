"""Command (Host to Tensor Bus) Definitions and State Structures."""

import msgspec


class BaseCommand(msgspec.Struct, tag=True):
    """Base class for all Tensor Bus commands.

    Features:
    - Auto-tagging: Uses class name as type tag
    - Common timestamp field for all commands
    """

    timestamp: float


class Send(BaseCommand):
    """Send tensor command."""

    pair_name: str


class Receive(BaseCommand):
    """Receive tensor command."""

    pair_name: str


class Register(BaseCommand):
    """Register handler command."""

    handler_id: str


class RegisterTensor(BaseCommand):
    # Note: This is for prototyping
    """Register a tensor for zero-copy sharing between processes.

    The tensor payload (pickled via PyTorch's ForkingPickler) is stored
    in LMDB at storage_key. The pickled payload contains all tensor
    metadata (shape, dtype, device, CUDA pointer, etc.).

    This message only contains the minimal metadata needed to locate
    and authorize access to the tensor.
    """

    tensor_id: str
    storage_key: str
    writer_pid: int


class RegisterPair(BaseCommand):
    """Register to a Pair (peer-to-peer communication endpoint).

    Design:
    - Pair is symmetric: both peers can send() or recv()
    - Similar to RDMA Queue Pair (QP)
    - Registration writes to TCPStore and polls until both peers are ready

    Args:
        pair_name: Unique identifier for this pair (e.g., "obs", "action")
        local_name: Name of local peer (e.g., "inference", "training")
        expected_world_size: Number of ranks for local peer
        remote_name: Name of remote peer (explicit pairing)
    """

    pair_name: str
    local_name: str
    expected_world_size: int
    remote_name: str


Message = Send | Receive | Register | RegisterTensor | RegisterPair
