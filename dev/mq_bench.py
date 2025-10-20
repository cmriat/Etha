# mq_bench.py
import os, sys, time
from multiprocessing.managers import BaseManager
from queue import Queue

MSG_SIZE = int(os.environ.get("MSG_SIZE", "1024"))
N = int(os.environ.get("N", "200000"))
ADDR = ("127.0.0.1", 50055)
AUTH = b"secret"

_q = Queue()
class M(BaseManager): ...
M.register("get_q", callable=lambda: _q)

def server():
    # 方式A（推荐）：前台阻塞式server
    m = M(address=ADDR, authkey=AUTH)
    srv = m.get_server()           # 不要调用 m.start()
    print(f"srv listening on {ADDR} ...")
    srv.serve_forever()

def _connect_manager(retries=100, pause=0.05):
    last_err = None
    for _ in range(retries):
        try:
            m = M(address=ADDR, authkey=AUTH)
            m.connect()
            return m
        except Exception as e:
            last_err = e
            time.sleep(pause)
    raise last_err

def prod():
    m = _connect_manager()
    q = m.get_q()
    payload = b"x" * MSG_SIZE
    t0 = time.perf_counter()
    for _ in range(N):
        q.put(payload)
    q.put(None)  # 结束标记
    t1 = time.perf_counter()
    print(f"[MPQ PROD] sent {N} msgs ({MSG_SIZE}B) in {t1-t0:.3f}s")

def cons():
    m = _connect_manager()
    q = m.get_q()
    c = 0
    t0 = time.perf_counter()
    while True:
        msg = q.get()
        if msg is None:
            break
        c += 1
    t1 = time.perf_counter()
    print(f"[MPQ CONS] recv {c} msgs in {t1-t0:.3f}s -> {c/(t1-t0):.0f} msg/s")

if __name__ == "__main__":
    role = sys.argv[1]
    {"server": server, "prod": prod, "cons": cons}[role]()
