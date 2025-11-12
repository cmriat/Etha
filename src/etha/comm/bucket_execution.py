"""Bucket communication executor."""

import logging
from collections import deque

import torch
import torch.distributed as dist

from .ir import Bucket, TransferType
from .utils import get_or_create_process_group
from .chunk_execution import _assemble_chunk, _prepare_recv_buffer, _prepare_send_buffer

logger = logging.getLogger(__name__)
rank = 0


def _prepare_source_bucket(bucket: Bucket) -> None:
    if len(bucket.offsets) == 1:
        logger.debug(f"[rank={rank}] Bucket execution: Preparing source bucket with one offset")
        _, length, chunk, _ = bucket.offsets[0]
        _prepare_send_buffer(chunk)
        bucket.buffer = chunk.buffer
        chunk.buffer = None
        return

    total_elems = bucket.offsets[-1][0] + bucket.offsets[-1][1]
    bucket.buffer = torch.empty(total_elems, dtype=bucket.dtype, device=bucket.device)

    for offset, length, chunk, _ in bucket.offsets:
        _prepare_send_buffer(chunk)
        flat = chunk.buffer.view(-1)
        bucket.buffer.narrow(0, offset, length).copy_(flat, non_blocking=True)
        chunk.buffer = None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(bucket.buffer.device))
    bucket.buffer_ready_event = event


def _prepare_target_bucket(bucket: Bucket) -> None:
    if len(bucket.offsets) == 1:
        logger.debug(f"[rank={rank}] Bucket execution: Preparing target bucket with one offset")
        _, length, chunk, _ = bucket.offsets[0]
        _prepare_recv_buffer(chunk)
        bucket.buffer = chunk.buffer
        return

    total_elems = bucket.offsets[-1][0] + bucket.offsets[-1][1]
    bucket.buffer = torch.empty(total_elems, dtype=bucket.dtype, device=bucket.device)

    for offset, length, chunk, shape in bucket.offsets:
        view = bucket.buffer.narrow(0, offset, length).view(shape)
        if bucket.transfer_type == TransferType.SELF_COPY:
            if chunk.tensor is None:
                raise ValueError("Self-copy chunk requires bound tensor.")
            view.copy_(chunk.tensor[chunk.src_slice_tuples], non_blocking=True)
        chunk.buffer = view
    if bucket.transfer_type == TransferType.SELF_COPY:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(bucket.buffer.device))
        bucket.buffer_ready_event = event


def _prepare_bucket(bucket: Bucket) -> None:
    if bucket.is_source:
        _prepare_source_bucket(bucket)
    else:
        _prepare_target_bucket(bucket)


def _bucket_key(bucket: Bucket) -> tuple:
    chunk = bucket.offsets[0][2]
    if bucket.is_source:
        return chunk.src_idx
    return chunk.dst_idx


def _launch_bucket(bucket: Bucket) -> bool:
    if bucket.buffer_ready_event is not None:
        if not bucket.buffer_ready_event.query():
            return False
        bucket.buffer_ready_event = None

    match bucket.transfer_type:
        case TransferType.SELF_COPY:
            bucket.work = None
            return True

        case TransferType.BROADCAST:
            if bucket.group_key is None:
                raise ValueError("Broadcast bucket missing process group key.")
            group_ranks = sorted([bucket.group_key[0], *bucket.group_key[1]])
            group = get_or_create_process_group(group_ranks)
            src_rank = bucket.src_rank if bucket.is_source else bucket.group_key[0]
            bucket.work = dist.broadcast(bucket.buffer, src=src_rank, group=group, async_op=True)

        case TransferType.P2P:
            if bucket.is_source:
                if not bucket.dst_ranks or len(bucket.dst_ranks) != 1:
                    raise ValueError("P2P bucket must have exactly one destination rank.")
                logger.debug(
                    f"[rank={rank}] Bucket execution: Launching P2P send to rank {bucket.dst_ranks[0]} buffer shape {bucket.buffer.shape} buffer dtype {bucket.buffer.dtype} buffer device {bucket.buffer.device}"
                )
                bucket.work = dist.isend(bucket.buffer, dst=bucket.dst_ranks[0])
            else:
                logger.debug(
                    f"[rank={rank}] Bucket execution: Launching P2P receive from rank {bucket.src_rank} buffer shape {bucket.buffer.shape} buffer dtype {bucket.buffer.dtype} buffer device {bucket.buffer.device}"
                )
                bucket.work = dist.irecv(bucket.buffer, src=bucket.src_rank)
    return True


def _is_bucket_complete(bucket: Bucket) -> bool:
    if bucket.work is None:
        return True
    if bucket.device is not None and bucket.device.type == "cpu":  # cpu device do not is_completed()
        bucket.work.wait()
        bucket.work = None
        return True
    return bucket.work.is_completed()


def _finalize_bucket(bucket: Bucket) -> None:
    if bucket.work is not None:
        bucket.work.wait()
        bucket.work = None

    if bucket.is_source:
        bucket.buffer = None
        return

    for _, _, chunk, _ in bucket.offsets:
        _assemble_chunk(chunk)

    bucket.buffer = None


def execute_bucket_pipeline(
    buckets: list[Bucket],
    max_in_flight: int,
) -> None:
    if not buckets:
        return
    global rank
    rank = dist.get_rank()

    key_states: dict[tuple, dict[str, deque]] = {}
    key_order: list[tuple] = []

    for bucket in buckets:
        key = _bucket_key(bucket)
        if key not in key_states:
            key_states[key] = {
                "candidate": deque(),
                "prepared": deque(),
                "in_flight": deque(),
            }
            key_order.append(key)
        key_states[key]["candidate"].append(bucket)
    logger.debug(f"[rank={rank}] Bucket execution: Key states: {key_states}")

    def _has_work() -> bool:
        for state in key_states.values():
            if state["candidate"] or state["prepared"] or state["in_flight"]:
                return True
        return False

    while _has_work():
        for key in key_order:
            state = key_states[key]
            candidate: deque = state["candidate"]
            prepared: deque = state["prepared"]
            in_flight: deque = state["in_flight"]

            while candidate and len(prepared) + len(in_flight) < max_in_flight:
                bucket = candidate.popleft()
                logger.debug(f"[rank={rank}] Bucket execution: Preparing bucket {bucket}")
                _prepare_bucket(bucket)
                prepared.append(bucket)

        for key in key_order:
            state = key_states[key]
            prepared: deque = state["prepared"]
            in_flight: deque = state["in_flight"]

            while prepared:
                if not _launch_bucket(prepared[0]):
                    break
                logger.debug(f"[rank={rank}] Bucket execution: Launching bucket {prepared[0]}")
                in_flight.append(prepared.popleft())

        for key in key_order:
            in_flight: deque = key_states[key]["in_flight"]
            while in_flight:
                if not _is_bucket_complete(in_flight[0]):
                    break
                logger.debug(f"[rank={rank}] Bucket execution: Finalizing bucket {in_flight[0]}")
                _finalize_bucket(in_flight.popleft())
