"""Integration test for the distributed model transfer prototype."""

import os
import time
import shutil
import signal
import socket
import subprocess
from pathlib import Path

import pytest


def _find_free_port() -> int:
    """Bind to an OS-assigned ephemeral port, close, return the number.

    There's still a TOCTOU window before the child re-binds, but four parallel
    ports collide with our long-running agent only when the agent's internal
    TCPStores grab the same number — picking from the ephemeral range moments
    before launch is good enough for a test harness.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _find_n_free_ports(n: int) -> tuple[int, ...]:
    """``_find_free_port`` ``n`` times, with within-call dedup.

    Back-to-back ``bind(0)`` calls *can* return the same port; the kernel makes
    it rare but does not promise uniqueness. Loop until we have ``n`` distinct.
    """
    selected: set[int] = set()
    while len(selected) < n:
        selected.add(_find_free_port())
    return tuple(selected)


def _spawn_process(cmd: list[str], env: dict[str, str], cwd: Path, log_path: Path) -> tuple[subprocess.Popen, object]:
    """Launch a subprocess with output redirected to a log file."""
    log_file = log_path.open("w", buffering=1)
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def _terminate_process(proc: subprocess.Popen, timeout: float = 30.0) -> None:
    """Terminate a subprocess gracefully, escalating if required."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    ("training_dtype", "inference_dtype"),
    [
        pytest.param("float32", "float32", id="float32_to_float32"),
        pytest.param("float32", "bfloat16", id="float32_to_bfloat16"),
    ],
)
def test_distributed_model_transfer_end_to_end(training_dtype: str, inference_dtype: str):
    project_root = Path(__file__).resolve().parents[1]

    logs_root = Path("./tests/distributed_model_transfer_logs")
    logs_dir = logs_root / f"{training_dtype}_to_{inference_dtype}"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    agent_log_path = logs_dir / "agent.log"
    train_log_path = logs_dir / "train.log"
    inference_log_path = logs_dir / "inference.log"

    # Clean LMDB directories to avoid stale state between runs.
    lmdb_root = Path("/tmp/dbs")
    lmdb_root.mkdir(parents=True, exist_ok=True)
    for candidate in lmdb_root.glob("*"):
        candidate.unlink(missing_ok=True)

    # Pick four free ephemeral ports per case to avoid collisions between cases
    # (agent's pair-handshake TCPStores grab ephemeral ports — a hard-coded
    # master-port for the next case can race with them).
    agent_port, train_port, inference_port, store_port = _find_n_free_ports(4)

    base_env = os.environ.copy()
    base_env.setdefault("PIXI_PROJECT_ROOT", str(project_root))
    base_env.setdefault("MASTER_ADDR", "127.0.0.1")
    base_env.setdefault("HF_HOME", "/data/hf")
    base_env.setdefault("HF_HUB_OFFLINE", "1")
    base_env["ETHA_STORE_PORT"] = str(store_port)

    processes: list[tuple[subprocess.Popen, object]] = []

    agent_cmd = [
        "torchrun",
        "--nproc_per_node=8",
        f"--master-port={agent_port}",
        "tests/distributed_model_transfer/agent.py",
    ]
    agent_proc = _spawn_process(agent_cmd, base_env, project_root, agent_log_path)
    processes.append(agent_proc)

    # Give agents time to initialize and register with the store.
    time.sleep(8)

    training_env = base_env.copy()
    training_env.setdefault("TRAINING_STRATEGY", "pure_mp")
    training_env.setdefault("MODEL_ID", "Qwen/Qwen3-0.6B")
    training_env["MODEL_DTYPE"] = training_dtype
    training_cmd = [
        "torchrun",
        "--nproc_per_node=4",
        f"--master-port={train_port}",
        "tests/distributed_model_transfer/train.py",
    ]
    train_proc = _spawn_process(training_cmd, training_env, project_root, train_log_path)
    processes.append(train_proc)

    inference_env = base_env.copy()
    inference_env.setdefault("INFERENCE_STRATEGY", "hybrid_dp_mp")
    inference_env.setdefault("AGENT_RANK_OFFSET", "4")
    inference_env.setdefault("MODEL_ID", "Qwen/Qwen3-0.6B")
    inference_env["MODEL_DTYPE"] = inference_dtype
    inference_cmd = [
        "torchrun",
        "--nproc_per_node=4",
        f"--master-port={inference_port}",
        "tests/distributed_model_transfer/inference.py",
    ]
    inference_proc = _spawn_process(inference_cmd, inference_env, project_root, inference_log_path)
    processes.append(inference_proc)

    try:
        # Wait for training and inference to complete, allowing ample time for model load.
        train_handle, train_log = train_proc
        inference_handle, inference_log = inference_proc
        train_handle.wait(timeout=3600)
        inference_handle.wait(timeout=3600)
        train_log.flush()
        inference_log.flush()

        assert train_handle.returncode == 0, train_log_path.read_text()
        assert inference_handle.returncode == 0, inference_log_path.read_text()

        inference_output = inference_log_path.read_text()
        assert "✅ Distributed model matches golden model" in inference_output
    finally:
        # Ensure all processes are cleaned up and log files are closed.
        for proc, log_file in reversed(processes):
            _terminate_process(proc)
            log_file.close()
        shutil.rmtree(logs_root, ignore_errors=True)
