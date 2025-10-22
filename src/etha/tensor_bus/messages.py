"""Message (Host to Tensor Bus) Definitions."""

import msgspec


class Command(msgspec.Struct, tag=True):
    """Base class for all Tensor Bus commands.

    Features:
    - Auto-tagging: Uses class name as type tag
    - Common timestamp field for all commands
    """

    timestamp: float


class Send(Command):
    """Send tensor command."""

    pair_name: str


class Receive(Command):
    """Receive tensor command."""

    pair_name: str


class Register(Command):
    """Register handler command."""

    handler_id: str


class RegisterTensor(Command):
    # TODO: A real prototyping
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


Message = Send | Receive | Register | RegisterTensor
