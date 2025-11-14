"""Bucket communication executor."""

import logging
from collections import deque, defaultdict

import torch
import torch.distributed as dist

from .ir import Bucket, TransferType
from .utils import get_or_create_process_group
from .chunk_execution import _prepare_chunk, _finalize_chunk

logger = logging.getLogger(__name__)
rank = None


def _prepare_source_bucket(bucket: Bucket) -> None:
    if len(bucket.entries) == 1:
        chunk = bucket.entries[0].chunk
        _prepare_chunk(chunk)
        bucket.buffer = chunk.buffer
        chunk.buffer = None
        return

    bucket.buffer = torch.empty(bucket.total_elems, dtype=bucket.dtype, device=bucket.device)

    for entry in bucket.entries:
        chunk = entry.chunk
        _prepare_chunk(chunk, contiguous=False)
        flat = chunk.buffer.view(-1)
        bucket.buffer.narrow(0, entry.offset, entry.numel).copy_(flat, non_blocking=True)
        chunk.buffer = None
    event = torch.cuda.Event()
    event.record()
    bucket.buffer_ready_event = event


def _prepare_target_bucket(bucket: Bucket) -> None:
    if len(bucket.entries) == 1:
        chunk = bucket.entries[0].chunk
        _prepare_chunk(chunk)
        bucket.buffer = chunk.buffer
        return

    bucket.buffer = torch.empty(bucket.total_elems, dtype=bucket.dtype, device=bucket.device)

    for entry in bucket.entries:
        chunk = entry.chunk
        view = bucket.buffer.narrow(0, entry.offset, entry.numel).view(chunk.chunk_shape)
        if bucket.transfer_type == TransferType.SELF_COPY:
            view.copy_(chunk.tensor[chunk.src_slice_tuples], non_blocking=True)
        chunk.buffer = view
    if bucket.transfer_type == TransferType.SELF_COPY:
        event = torch.cuda.Event()
        event.record()
        bucket.buffer_ready_event = event


def _prepare_bucket(bucket: Bucket) -> None:
    if bucket.is_source:
        _prepare_source_bucket(bucket)
    else:
        _prepare_target_bucket(bucket)


def _launch_bucket(bucket: Bucket) -> bool:
    if bucket.buffer_ready_event is not None:
        if not bucket.buffer_ready_event.query():
            return False
        bucket.buffer_ready_event = None

    match bucket.transfer_type:
        case TransferType.SELF_COPY:
            bucket.work = None

        case TransferType.BROADCAST:
            group_ranks = sorted([bucket.src_rank, *bucket.dst_ranks])
            group = get_or_create_process_group(group_ranks)
            bucket.work = dist.broadcast(bucket.buffer, src=bucket.src_rank, group=group, async_op=True)

        case TransferType.P2P:
            if bucket.is_source:
                bucket.work = dist.isend(bucket.buffer, dst=bucket.dst_ranks[0])
            else:
                bucket.work = dist.irecv(bucket.buffer, src=bucket.src_rank)
    return True


def _is_bucket_complete(bucket: Bucket) -> bool:
    if bucket.work is None:
        return True
    if bucket.device is not None and bucket.device.type == "cpu":  # cpu device do not support is_completed()
        bucket.work.wait()
        bucket.work = None
        return True
    return bucket.work.is_completed()


def _finalize_bucket(bucket: Bucket) -> list[torch.cuda.Event]:
    events: list[torch.cuda.Event] = []
    if bucket.work is not None:
        bucket.work.wait()
        bucket.work = None

    if not bucket.is_source:
        for entry in bucket.entries:
            event = _finalize_chunk(entry.chunk)
            if event is not None:
                events.append(event)

    bucket.buffer = None

    return events


def execute_bucket_pipeline(
    buckets: list[Bucket],
    max_in_flight: int = 2,
) -> None:
    global rank
    rank = dist.get_rank()

    key_states: defaultdict[tuple, dict[str, deque]] = defaultdict(
        lambda: {
            "candidate": deque(),
            "prepared": deque(),
            "in_flight": deque(),
        }
    )
    for bucket in buckets:
        key_states[bucket.key]["candidate"].append(bucket)
    logger.debug(f"[rank={rank}] Bucket execution: Key states: {key_states}")

    def _has_work() -> bool:
        for state in key_states.values():
            if state["candidate"] or state["prepared"] or state["in_flight"]:
                return True
        return False

    events: list[torch.cuda.Event] = []
    while _has_work():
        for state in key_states.values():
            candidate: deque = state["candidate"]
            prepared: deque = state["prepared"]
            in_flight: deque = state["in_flight"]

            while candidate and len(prepared) + len(in_flight) < max_in_flight:
                bucket = candidate.popleft()
                logger.debug(f"[rank={rank}] Bucket execution: Preparing bucket {bucket}")
                _prepare_bucket(bucket)
                prepared.append(bucket)

        for state in key_states.values():
            prepared: deque = state["prepared"]
            in_flight: deque = state["in_flight"]

            while prepared:
                if not _launch_bucket(prepared[0]):
                    break
                logger.debug(f"[rank={rank}] Bucket execution: Launching bucket {prepared[0]}")
                in_flight.append(prepared.popleft())

        for state in key_states.values():
            in_flight: deque = state["in_flight"]
            while in_flight:
                if not _is_bucket_complete(in_flight[0]):
                    break
                logger.debug(f"[rank={rank}] Bucket execution: Finalizing bucket {in_flight[0]}")
                events.extend(_finalize_bucket(in_flight.popleft()))

    for event in events:
        event.wait()
