"""trainer 做成 vLLM 形状的 server:对外一个 HTTP 入口,内部 collective 扇出走 zmq
——与 vLLM 的 api-server(HTTP)→ workers(zmq)同构。

rank0 起 FastAPI 前端线程;各 rank(含 rank0)在主线程跑 zmq REP loop 执行方法
(NCCL 调用线程稳定)。前端对全 rank 先 send 后 recv:collective 调用
(create_cross_group / chunk_comm)必须全 rank 并发进入。payload 全程显式
pickle bytes(tensor 按值,不走 fd 共享)。
"""

import pickle
import threading

HTTP_PORT = 52100
ZMQ_PORT_BASE = 52200


def serve(obj, rank, world):
    import zmq

    rep = zmq.Context().socket(zmq.REP)
    rep.bind(f"tcp://*:{ZMQ_PORT_BASE + rank}")
    if rank == 0:
        threading.Thread(target=_http_frontend, args=(world,), daemon=True).start()
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


def _http_frontend(world):
    import uvicorn
    import zmq
    from fastapi import FastAPI, Request, Response

    ctx = zmq.Context()
    reqs = []
    for r in range(world):
        sock = ctx.socket(zmq.REQ)
        sock.connect(f"tcp://127.0.0.1:{ZMQ_PORT_BASE + r}")
        reqs.append(sock)
    app = FastAPI()

    @app.post("/rpc")
    async def rpc(request: Request):
        payload = await request.body()
        for sock in reqs:
            sock.send(payload)
        outs = [pickle.loads(sock.recv()) for sock in reqs]
        return Response(pickle.dumps(outs), media_type="application/octet-stream")

    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="warning")
