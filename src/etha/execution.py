"""Run a bucket plan: 同 (src_rank, dst_ranks, transport, 层) 的 chunk 聚成一个
``Bucket``,一次 broadcast/P2P 替 per-chunk —— 消掉 per-chunk launch 间隙(通信 gap)。

per-channel pipeline(prepare 拼大 buffer → launch collective → finalize 切回/feed):
``buffer_ready_event`` 让 prepare(GPU copy)与在飞 collective(NCCL)重叠。流式内存:
bucket 的 dst chunk 按需分配(``target_alloc``),一桶收齐即逐 weight 喂 loader 并释放
(``on_complete``)——峰值 = max_in_flight × bucket 大小。experts 桶限层不跨(撞 vLLM
layerwise reload 逐层 process 上界),replicate 可跨层(占比小)。
"""

import os
import re
import time
from collections import deque, defaultdict
from collections.abc import Callable

import torch
import torch.distributed as dist

from .ir import Chunk, Bucket, Transport

_PROGRESS = int(os.environ.get("ETHA_PROGRESS", "0"))
_INFLIGHT = int(os.environ.get("ETHA_INFLIGHT", "4"))
_PROFILE_ROUND = int(os.environ.get("ETHA_PROFILE_ROUND", "1"))
_profile_calls = [0]


def _layer(weight: str | None) -> int:
    m = re.search(r"layers\.(\d+)", weight or "")
    return int(m.group(1)) if m else -1


def chunk_comm(
    chunks: list[Chunk],
    group: dist.ProcessGroup | None = None,
    subgroups: dict[tuple[int, ...], dist.ProcessGroup] | None = None,
    max_in_flight: int = _INFLIGHT,
    target_alloc: Callable[[str], torch.Tensor] | None = None,
    on_complete: Callable[[str, torch.Tensor], None] | None = None,
) -> None:
    pending: dict[str, int] = defaultdict(int)
    allocated: dict[str, torch.Tensor] = {}
    sample = next((t for c in chunks for t in (c.src_tensor, c.dst_tensor) if t is not None), None)
    use_event = (sample is not None and sample.device.type == "cuda") or (
        group is not None and dist.get_backend(group) == "nccl"
    )  # vLLM 收端聚合前 dst_tensor 未 alloc→sample=None,得靠 group backend 判 GPU,否则走 work.wait() 同步、done 的 DtoD copy 不 overlap 通信

    def is_dst(chunk: Chunk) -> bool:
        return len(chunk.dst_slice) > 0

    def alloc_dst(chunk: Chunk) -> None:
        if chunk.dst_tensor is None and is_dst(chunk) and target_alloc is not None:
            if chunk.weight not in allocated:
                allocated[chunk.weight] = target_alloc(chunk.weight)
            chunk.dst_tensor = allocated[chunk.weight]

    def feed(chunk: Chunk) -> None:
        if is_dst(chunk) and chunk.weight is not None:
            pending[chunk.weight] -= 1
            if pending[chunk.weight] == 0 and on_complete is not None:
                on_complete(chunk.weight, allocated.pop(chunk.weight, chunk.dst_tensor))

    locals_: list[Chunk] = []
    wire: list[Chunk] = []
    for chunk in chunks:
        if is_dst(chunk) and chunk.weight is not None:
            pending[chunk.weight] += 1
        (locals_ if chunk.transport == Transport.LOCAL else wire).append(chunk)

    for chunk in locals_:  # 本地自拷:读 src → 写 dst,无 wire
        alloc_dst(chunk)
        chunk.prepare()
        chunk.finalize()
        feed(chunk)

    # 聚合:同 (src, dst, transport, 层) 的 chunk 按 route 序进一个 bucket。route 序两端
    # 一致 → P2P 收发配对、broadcast 子组同序。
    groups: defaultdict[tuple, list[Chunk]] = defaultdict(list)
    for chunk in sorted(wire, key=lambda c: c.route_idx):
        groups[(chunk.src_rank, chunk.dst_ranks, chunk.transport, _layer(chunk.weight))].append(chunk)
    buckets = [
        Bucket(chunks=cs, transport=cs[0].transport, src_rank=cs[0].src_rank, dst_ranks=cs[0].dst_ranks)
        for cs in groups.values()
    ]

    def bind(bucket: Bucket) -> None:
        for chunk in bucket.chunks:
            alloc_dst(chunk)
        bucket.prepare()

    def done(bucket: Bucket) -> None:
        layer = _layer(bucket.chunks[0].weight)
        with torch.profiler.record_function(f"fin_copy:L{layer}"):  # bucket.finalize:DtoD 切回 dst_tensor
            bucket.finalize()
        with torch.profiler.record_function(f"feed:L{layer}"):  # on_complete → vLLM load_weights(materialize/process)
            for chunk in bucket.chunks:
                feed(chunk)

    def launch(bucket: Bucket) -> dist.Work | None:
        with torch.profiler.record_function(f"comm:{bucket.transport.name}:L{_layer(bucket.chunks[0].weight)}"):
            if bucket.transport == Transport.BROADCAST:
                key = tuple(sorted({bucket.src_rank, *bucket.dst_ranks}))
                opts = dist.BroadcastOptions()
                opts.rootRank = key.index(bucket.src_rank)
                opts.rootTensor = 0
                return subgroups[key].broadcast([bucket.buffer], opts)
            if bucket.chunks[0].src_tensor is not None:
                return dist.isend(bucket.buffer, bucket.dst_ranks[0], group=group)
            return dist.irecv(bucket.buffer, bucket.src_rank, group=group)

    def is_done(work: dist.Work | None) -> bool:  # gloo is_completed CPU 永 False→退 wait;GPU 非阻塞。launch-all-before-finalize 避死锁
        if work is None:
            return True
        if not use_event:
            work.wait()
            return True
        return work.is_completed()

    profile_path = os.environ.get("ETHA_PROFILE")
    do_profile = bool(profile_path) and _profile_calls[0] == _PROFILE_ROUND
    _profile_calls[0] += 1
    prof = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,  # 全程 profile(无 schedule),靠 record_function 标 comm/fin_copy/feed 段;无 stack 故 trace 小
        )
        if do_profile
        else None
    )
    if prof is not None:
        prof.start()

    t0 = time.perf_counter()
    total = len(buckets)
    finished = [0]

    channels: defaultdict[tuple, dict[str, deque]] = defaultdict(
        lambda: {"cand": deque(), "prep": deque(), "fly": deque()}
    )
    for bucket in buckets:
        channels[(bucket.src_rank, bucket.dst_ranks, bucket.transport)]["cand"].append(bucket)

    def has_work() -> bool:
        return any(c["cand"] or c["prep"] or c["fly"] for c in channels.values())

    while has_work():
        progress = True
        while progress:  # prepare:拼 bucket buffer 到 max_in_flight(GPU copy 异步,event 记 ready)
            progress = False
            for ch in channels.values():
                if ch["cand"] and len(ch["prep"]) + len(ch["fly"]) < max_in_flight:
                    bucket = ch["cand"].popleft()
                    bind(bucket)
                    ev = torch.cuda.Event() if use_event else None
                    if ev is not None:
                        ev.record()
                    ch["prep"].append((bucket, ev))
                    progress = True
        progress = True
        while progress:  # launch:buffer-ready 的全发出去——再 finalize 才能 P2P 全配对、不死锁
            progress = False
            for ch in channels.values():
                if ch["prep"] and (ch["prep"][0][1] is None or ch["prep"][0][1].query()):
                    bucket, _ = ch["prep"].popleft()
                    ch["fly"].append((bucket, launch(bucket)))
                    progress = True
        progress = True
        while progress:  # finalize:完成的切回/feed(此时 work 已全 launch、后台配对中,CPU wait 不死锁)
            progress = False
            for ch in channels.values():
                if ch["fly"] and is_done(ch["fly"][0][1]):
                    bucket, _ = ch["fly"].popleft()
                    done(bucket)
                    finished[0] += 1
                    progress = True
                    if prof is not None:
                        prof.step()
                    if _PROGRESS and finished[0] % _PROGRESS == 0:
                        print(f"[chunk_comm] {finished[0]}/{total} {time.perf_counter() - t0:.1f}s", flush=True)

    if use_event:
        torch.cuda.synchronize()

    if prof is not None:
        prof.stop()
        if on_complete is not None and dist.get_rank() == 0:  # 只导收端(vLLM)rank0:全 rank profile 保对称,只 1 文件免塞家目录
            prof.export_chrome_trace(f"{profile_path}.{__import__('socket').gethostname()}_r{dist.get_rank()}.json")
