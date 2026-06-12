"""trainer 侧 collective RPC over HTTP(FastAPI):对齐 vLLM collective_rpc 的形状。

payload 是显式 pickle bytes(按值序列化,tensor 不走 fd 共享);driver 线程池
并发 POST 全体 rank——collective 调用(create_cross_group / chunk_comm)必须
全 rank 同时进入。async handler 在主 event loop 线程执行,NCCL 调用线程稳定。
"""

import pickle
import time
from concurrent.futures import ThreadPoolExecutor


def serve(obj, port):
    import uvicorn
    from fastapi import FastAPI, Request, Response

    app = FastAPI()

    @app.post("/rpc")
    async def rpc(request: Request):
        method, args, kwargs = pickle.loads(await request.body())
        if method == "ping":
            out = ("ok", None)
        else:
            try:
                out = ("ok", getattr(obj, method)(*args, **kwargs))
            except Exception as e:
                out = ("err", f"{type(e).__name__}: {e}")
        return Response(pickle.dumps(out), media_type="application/octet-stream")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


class CollectiveClient:
    def __init__(self, addrs, retry=180):
        import requests

        self.requests = requests
        self.urls = [f"http://{host}:{port}/rpc" for host, port in addrs]
        self.pool = ThreadPoolExecutor(len(self.urls))
        ping = pickle.dumps(("ping", (), {}))
        for url in self.urls:
            for _ in range(retry):
                try:
                    self.requests.post(url, data=ping, timeout=5)
                    break
                except self.requests.exceptions.ConnectionError:
                    time.sleep(2)

    def collective_rpc_async(self, method, *args, **kwargs):
        payload = pickle.dumps((method, args, kwargs))
        futs = [self.pool.submit(self.requests.post, url, data=payload, timeout=3600) for url in self.urls]

        def wait():
            outs = []
            for f in futs:
                status, val = pickle.loads(f.result().content)
                if status == "err":
                    raise RuntimeError(val)
                outs.append(val)
            return outs

        return wait

    def collective_rpc(self, method, *args, **kwargs):
        return self.collective_rpc_async(method, *args, **kwargs)()
