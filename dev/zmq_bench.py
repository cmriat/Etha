# zmq_bench.py  运行：python zmq_bench.py prod &  python zmq_bench.py cons
import os, sys, time, zmq

SOCK = "ipc:///tmp/bench.sock"
MSG_SIZE = int(os.environ.get("MSG_SIZE", "1024"))      # 字节
N = int(os.environ.get("N", "200000"))

def prod():
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PUSH)
    s.setsockopt(zmq.SNDHWM, 100000)
    s.bind(SOCK)
    msg = b"x" * MSG_SIZE
    t0 = time.perf_counter()
    for _ in range(N):
        s.send(msg, copy=False)  # 避免一次用户态拷贝
    s.send(b"", copy=False)      # 终止标记
    t1 = time.perf_counter()
    print(f"[ZMQ PROD] sent {N} msgs ({MSG_SIZE}B) in {t1-t0:.3f}s")

def cons():
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PULL)
    s.setsockopt(zmq.RCVHWM, 100000)
    s.connect(SOCK)
    c = 0
    t0 = time.perf_counter()
    while True:
        m = s.recv()
        if not m: break
        c += 1
    t1 = time.perf_counter()
    print(f"[ZMQ CONS] recv {c} msgs in {t1-t0:.3f}s -> {c/(t1-t0):.0f} msg/s")

if __name__ == "__main__":
    if sys.argv[1] == "prod": prod()
    else: cons()
