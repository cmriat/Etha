"""trainer 做成 vLLM 形状的 server:对外一个 HTTP 入口,内部 collective 扇出走 zmq
——与 vLLM 的 api-server(HTTP)→ workers(zmq)同构。

rank0 起 FastAPI 前端线程;各 rank(含 rank0)在主线程跑 zmq REP loop 执行方法
(NCCL 调用线程稳定)。前端对全 rank 先 send 后 recv:collective 调用
(create_cross_group / chunk_comm)必须全 rank 并发进入。payload 全程显式
pickle bytes(tensor 按值,不走 fd 共享)。
"""

import os
import pickle
import threading

HTTP_PORT = int(os.environ.get("ETHA_HTTP_PORT", 52100))
ZMQ_PORT_BASE = HTTP_PORT + 100


def serve(obj, rank, world, rank_ips=None):
    # rank_ips[r] = rank r 所在节点的 IP(多节点;None=单节点全 127.0.0.1)。
    import time

    import zmq

    rep = zmq.Context().socket(zmq.REP)
    addr = f"tcp://*:{ZMQ_PORT_BASE + rank}"
    for _ in range(40):  # 残留 job 进程 teardown 有延迟,等它释放端口(hostNetwork 共享)
        try:
            rep.bind(addr)
            break
        except zmq.error.ZMQError:
            time.sleep(3)
    else:
        rep.bind(addr)
    if rank == 0:
        ips = rank_ips or ["127.0.0.1"] * world
        threading.Thread(target=_http_frontend, args=(world, ips), daemon=True).start()
    while True:
        method, args, kwargs = pickle.loads(rep.recv())
        if method == "ping":
            out = ("ok", None)
        else:
            try:
                out = ("ok", getattr(obj, method)(*args, **kwargs))
            except Exception as e:
                out = ("err", f"{type(e).__name__}: {e}")
        rep.send(pickle.dumps(out))


def _http_frontend(world, rank_ips):
    import uvicorn
    import zmq
    from fastapi import FastAPI, HTTPException, Request, Response

    ctx = zmq.Context()
    reqs = []
    for r in range(world):
        sock = ctx.socket(zmq.REQ)
        sock.connect(f"tcp://{rank_ips[r]}:{ZMQ_PORT_BASE + r}")  # 跨节点走真实 IP
        reqs.append(sock)
    app = FastAPI()

    @app.post("/{method}")
    async def call(method: str, request: Request):
        body = await request.body()
        args, kwargs = pickle.loads(body) if body else ((), {})
        payload = pickle.dumps((method, args, kwargs))
        for sock in reqs:
            sock.send(payload)
        outs = [pickle.loads(sock.recv()) for sock in reqs]
        for rank, (status, val) in enumerate(outs):
            if status == "err":
                raise HTTPException(status_code=500, detail=f"rank {rank}: {val}")
        return Response(pickle.dumps([val for _, val in outs]), media_type="application/octet-stream")

    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="warning")
