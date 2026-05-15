"""Single-node end-to-end launcher: agents, trainer, vLLM server, chat client.

Knobs live on `common.CONFIG`. The launcher only sets env vars the
underlying frameworks demand before fork.
"""

import os
import sys
import time
import shutil
import signal
import socket
import logging
import subprocess
from pathlib import Path

from common import CONFIG

HERE = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)


def _wait_for_port(host: str, port: int, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"agent TCPStore at {host}:{port} did not come up within {timeout}s")


def _spawn(
    name: str,
    args: list[str],
    extra_env: dict[str, str],
    log_root: Path,
    procs: list[subprocess.Popen],
) -> None:
    log_path = log_root / f"{name}.log"
    f = log_path.open("w")
    logger.info("==> %s -> %s", name, log_path)
    env = os.environ.copy()
    env.update(extra_env)
    p = subprocess.Popen(
        args,
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=str(HERE),
        start_new_session=True,  # own process group for clean SIGINT/SIGKILL
    )
    procs.append(p)


def _cleanup(procs: list[subprocess.Popen], grace_seconds: float = 15.0) -> None:
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and any(p.poll() is None for p in procs):
        time.sleep(0.5)
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [launch] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    log_root = HERE / "logs"
    shutil.rmtree(CONFIG.lmdb_root, ignore_errors=True)
    shutil.rmtree(log_root, ignore_errors=True)
    Path(CONFIG.lmdb_root).mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "topology: trainer=%d (attn:r=%d×s=%d, moe:r=%d×s=%d×ep=%d) vllm=%d (DP=%d TP=%d)",
        CONFIG.trainer_world_size,
        CONFIG.trainer_attn_dp_replicate,
        CONFIG.trainer_attn_dp_shard,
        CONFIG.trainer_moe_dp_replicate,
        CONFIG.trainer_moe_dp_shard,
        CONFIG.trainer_ep_size,
        CONFIG.vllm_world_size,
        CONFIG.vllm_dp_size,
        CONFIG.vllm_tp_size,
    )
    logger.info("model: %s", CONFIG.model_id)

    procs: list[subprocess.Popen] = []

    def shutdown_handler(*_):
        logger.info("signal received, shutting down children")
        _cleanup(procs)
        sys.exit(130)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    _spawn(
        "agent",
        ["torchrun", f"--nproc_per_node={CONFIG.agent_world_size}", "--master-port=49500", str(HERE / "agent.py")],
        {},
        log_root,
        procs,
    )

    logger.info("waiting for agent TCPStore at %s:%d ...", CONFIG.store_host, CONFIG.store_port)
    _wait_for_port(CONFIG.store_host, CONFIG.store_port, timeout=120.0)
    logger.info("agents up")

    trainer_cuda = ",".join(str(i) for i in range(CONFIG.trainer_world_size))
    _spawn(
        "trainer",
        ["torchrun", f"--nproc_per_node={CONFIG.trainer_world_size}", "--master-port=49501", str(HERE / "trainer.py")],
        {"CUDA_VISIBLE_DEVICES": trainer_cuda, "AGENT_RANK_OFFSET": "0"},
        log_root,
        procs,
    )

    vllm_cuda = ",".join(str(i) for i in range(CONFIG.trainer_world_size, CONFIG.agent_world_size))
    _spawn(
        "vllm",
        [sys.executable, str(HERE / "vllm_server.py")],
        {"CUDA_VISIBLE_DEVICES": vllm_cuda},
        log_root,
        procs,
    )

    _spawn("chat", [sys.executable, str(HERE / "chat_client.py")], {}, log_root, procs)

    logger.info("logs in %s; tail -f <log> to watch", log_root)

    rc = procs[1].wait()
    logger.info("trainer exited rc=%d, shutting down others", rc)
    _cleanup(procs)
    sys.exit(rc)


if __name__ == "__main__":
    main()
