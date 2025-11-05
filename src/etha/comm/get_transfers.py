"""Convert mesh-to-mesh communication maps to Transfer IR."""

from torch.distributed import ProcessGroup
from torch.distributed.tensor import Placement, DeviceMesh

from etha.comm.ir import Transfer, TransferType


def _assign_pipeline_stages(
    transfers: list[Transfer],
    num_transfer_per_stage: int = 4,
) -> None:
    """Assign pipeline stage IDs to transfers for pipelined execution.

    Args:
        transfers: List of Transfer objects to assign stages to
        num_transfer_per_stage: Number of operations per pipeline stage
    """
    for i, transfer in enumerate(transfers):
        transfer.stage_id = i // num_transfer_per_stage


def map_to_transfers(
    m2m_map: dict[int, dict[tuple, list[tuple[int, tuple]]]],
    source_num_slicers: list[int],
    target_num_slicers: list[int],
) -> list[Transfer]:
    """Convert source to taret map to Transfer IR with complete metadata.

    Args:
        m2m_map: src_rank -> {src_idx: [(dst_rank, dst_idx), ...]}
        source_num_slicers: Partitioning of source tensor
        target_num_slicers: Partitioning of target tensor

    Returns:
        List of Transfer objects with complete metadata (excluding self-copy)
    """
    transfers = []
    for src_rank in sorted(m2m_map.keys()):
        for src_idx in sorted(m2m_map[src_rank].keys()):
            dst_list = m2m_map[src_rank][src_idx]

            other_dsts = [(r, idx) for r, idx in dst_list if r != src_rank]

            if other_dsts:
                # Determine transfer type based on number of destinations
                transfer_type = TransferType.BROADCAST if len(other_dsts) > 1 else TransferType.P2P

                transfers.append(
                    Transfer(
                        src_rank=src_rank,
                        src_idx=src_idx,
                        dst_list=other_dsts,
                        transfer_type=transfer_type,
                        source_num_slicers=source_num_slicers,
                        target_num_slicers=target_num_slicers,
                    )
                )

    _assign_pipeline_stages(transfers)

    return transfers


def get_m2m_transfers(
    source_mesh: DeviceMesh,
    source_placements: tuple[Placement, ...],
    target_mesh: DeviceMesh,
    target_placements: tuple[Placement, ...],
    group: ProcessGroup,
    device: str = "cpu",
) -> list[Transfer]:
    """Generate Transfer IR for mesh-to-mesh communication.

    This is the primary user-facing function for generating a complete communication
    plan. Each Transfer contains all necessary metadata including partitioning info.

    Args:
        source_mesh: Source device mesh
        source_placements: Source tensor placements
        target_mesh: Target device mesh
        target_placements: Target tensor placements
        group: Process group for communication
        device: Device for execution ("cpu" or "cuda")

    Returns:
        List of Transfer objects with complete metadata
    """
    from .get_m2m_map import get_m2m_map

    # Generate maps
    m2m_map, source_num_slicers, target_num_slicers = get_m2m_map(
        source_mesh=source_mesh,
        source_placements=source_placements,
        target_mesh=target_mesh,
        target_placements=target_placements,
        group=group,
        device=device,
    )

    # Convert to Transfer IR
    return map_to_transfers(m2m_map, source_num_slicers, target_num_slicers)
