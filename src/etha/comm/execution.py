"""Communication executor."""

import logging
from collections import deque, defaultdict

import torch
import torch.distributed as dist

from .ir import Chunk, Bucket

logger = logging.getLogger(__name__)


def execute_bucket_pipeline(
    buckets: list[Bucket],
    max_in_flight: int = 2,
) -> None:
    rank = dist.get_rank()

    channels: defaultdict[tuple, dict[str, deque]] = defaultdict(
        lambda: {
            "candidate": deque(),
            "prepared": deque(),
            "in_flight": deque(),
        }
    )
    for bucket in buckets:
        channels[bucket.key]["candidate"].append(bucket)
    logger.debug(f"[rank={rank}] Bucket execution: Key states: {channels}")

    def _has_work() -> bool:
        for channel in channels.values():
            if channel["candidate"] or channel["prepared"] or channel["in_flight"]:
                return True
        return False

    while _has_work():
        for channel in channels.values():
            candidate: deque = channel["candidate"]
            prepared: deque = channel["prepared"]
            in_flight: deque = channel["in_flight"]

            while candidate and len(prepared) + len(in_flight) < max_in_flight:
                bucket = candidate.popleft()
                logger.debug(f"[rank={rank}] Bucket execution: Preparing bucket {bucket}")
                bucket.prepare()
                prepared.append(bucket)

        for channel in channels.values():
            prepared: deque = channel["prepared"]
            in_flight: deque = channel["in_flight"]

            while prepared:
                if not prepared[0].launch():
                    break
                logger.debug(f"[rank={rank}] Bucket execution: Launching bucket {prepared[0]}")
                in_flight.append(prepared.popleft())

        for channel in channels.values():
            in_flight: deque = channel["in_flight"]
            while in_flight:
                if not in_flight[0].is_complete():
                    break
                logger.debug(f"[rank={rank}] Bucket execution: Finalizing bucket {in_flight[0]}")
                in_flight.popleft().finalize()

    torch.cuda.synchronize()


def execute_chunk_simple(
    chunks: list[Chunk],
) -> None:
    """Execute chunks using polymorphic prepare/finalize and transfer functions.

    This is the simple sequential execution strategy - each chunk is fully
    prepared, launched, and finalized before moving to the next chunk.
    """
    for chunk in chunks:
        if chunk.tensor is None:
            continue
        chunk.prepare()
        chunk.work = chunk.execute()
        chunk.finalize()
        torch.cuda.synchronize()
