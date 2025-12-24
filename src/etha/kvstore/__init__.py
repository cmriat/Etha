"""KV Store abstraction for distributed coordination."""

import os
from typing import Literal

from .tcp import TorchTCPStore
from .base import KVStore
from .etcd import EtcdStore


def create_store(
    host: str = "localhost",
    port: int | None = None,
    backend: Literal["etcd", "tcp"] = "tcp",
    timeout: float = 3600.0,
    namespace: str = "default",
    component: str = "tensorbus",
) -> KVStore:
    """Create a KVStore instance.

    Args:
        host: Server host
        port: Server port (default: 2379 for etcd, 29500 for tcp)
        backend: "etcd" or "tcp"
        timeout: Connection timeout in seconds
        namespace: Namespace for key isolation
        component: Default component name

    Returns:
        KVStore instance
    """
    rank = int(os.environ.get("RANK", 0))

    if backend == "etcd":
        # Only rank 0 cleans up stale keys
        return EtcdStore(
            host=host,
            port=port or 2379,
            timeout=timeout,
            cleanup=False,
            namespace=namespace,
            component=component,
        )
    else:
        # TCP store uses environment variables for distributed setup
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        return TorchTCPStore(
            host=host,
            port=port or 29500,
            world_size=world_size,
            is_master=(rank == 0),
            timeout=timeout,
            wait_for_workers=True,
            namespace=namespace,
            component=component,
        )


__all__ = ["KVStore", "EtcdStore", "TorchTCPStore", "create_store"]
