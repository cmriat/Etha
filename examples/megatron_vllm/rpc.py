"""trainer 侧的最小 collective RPC:对齐 vLLM collective_rpc 的形状。

每个 trainer rank 一个 Listener;driver 先对全体 send 再统一 recv——
collective 调用(create_cross_group / chunk_comm)必须全 rank 并发进入。
标准库 multiprocessing.connection,pickle 自动。
"""

import time
from multiprocessing.connection import Client, Listener

AUTHKEY = b"etha"


def serve(obj, port):
    with Listener(("0.0.0.0", port), authkey=AUTHKEY) as listener:
        with listener.accept() as conn:
            while True:
                try:
                    method, args, kwargs = conn.recv()
                except EOFError:
                    return
                try:
                    conn.send(("ok", getattr(obj, method)(*args, **kwargs)))
                except Exception as e:
                    conn.send(("err", f"{type(e).__name__}: {e}"))


class CollectiveClient:
    def __init__(self, addrs, retry=120):
        self.conns = []
        for addr in addrs:
            for _ in range(retry):
                try:
                    self.conns.append(Client(addr, authkey=AUTHKEY))
                    break
                except ConnectionRefusedError:
                    time.sleep(2)

    def send_all(self, method, *args, **kwargs):
        for c in self.conns:
            c.send((method, args, kwargs))

    def recv_all(self):
        outs = []
        for c in self.conns:
            status, val = c.recv()
            if status == "err":
                raise RuntimeError(val)
            outs.append(val)
        return outs

    def collective_rpc(self, method, *args, **kwargs):
        self.send_all(method, *args, **kwargs)
        return self.recv_all()
