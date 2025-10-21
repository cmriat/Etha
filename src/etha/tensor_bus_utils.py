"""Utility classes for Tensor Bus middleware."""

from __future__ import annotations

from typing import List


class State:
    """State management for tensor transfer coordination."""

    def __init__(self, state: bool = False, target_ranks: List[int] = None):
        """Initialize state.
        
        Args:
            state: Current state (True = busy, False = ready)
            target_ranks: List of target rank IDs for distributed transfer
        """
        self.state = state
        self.target_ranks = target_ranks if target_ranks is not None else []


class RPC:
    """RPC client for communicating with middleware service."""

    def __init__(self, addr: str):
        """Initialize RPC client.
        
        Args:
            addr: Address of the middleware service
        """
        self.addr = addr

    def get_state(self) -> bool:
        """Get current state from middleware.
        
        Returns:
            bool: Current state value
        """
        # Placeholder for actual RPC implementation
        # In production, this would make an HTTP request or use another IPC mechanism
        raise NotImplementedError("RPC implementation required")

    def set_state(self, state: bool) -> None:
        """Set state in middleware.
        
        Args:
            state: New state value to set
        """
        # Placeholder for actual RPC implementation
        raise NotImplementedError("RPC implementation required")

    def put(self) -> None:
        """Signal that new data is available."""
        # Placeholder for actual RPC implementation
        raise NotImplementedError("RPC implementation required")


class InferServer:
    """Client for communicating with inference server."""

    def __init__(self, addr: str):
        """Initialize inference server client.
        
        Args:
            addr: Address of the inference server
        """
        self.addr = addr

    def prepare_recv(self, target_ranks: List[int] = None) -> None:
        """Prepare inference server to receive new weights.
        
        Args:
            target_ranks: Optional list of target ranks for distributed setup
        """
        # Placeholder for actual implementation
        raise NotImplementedError("InferServer implementation required")

    def get_state(self) -> bool:
        """Get current state of inference server.
        
        Returns:
            bool: Current state
        """
        # Placeholder for actual implementation
        raise NotImplementedError("InferServer implementation required")
