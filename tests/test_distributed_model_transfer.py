"""Integration test for the distributed model transfer prototype.

This test mirrors the launch script so it exercises the full Qwen3-30B hand-off
across two 4-GPU workers. It is disabled by default; set
RUN_DISTRIBUTED_MODEL_TRANSFER_TEST=1 when the required hardware, offline model
cache, and agent infrastructure are available.
"""

from __future__ import annotations

import os
import time
import shutil
import signal
import subprocess
from pathlib import Path

import pytest


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
def test_distributed_model_transfer_end_to_end():
    project_root = Path(__file__).resolve().parents[1]

    logs_dir = Path("./tests/distributed_model_transfer_logs")
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

    base_env = os.environ.copy()
    base_env.setdefault("PIXI_PROJECT_ROOT", str(project_root))
    base_env.setdefault("MASTER_ADDR", "127.0.0.1")
    base_env.setdefault("HF_HOME", "/data/hf")
    base_env.setdefault("HF_HUB_OFFLINE", "1")

    processes: list[tuple[subprocess.Popen, object]] = []

    agent_cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "--master-port=39500",
        "prototyping/agent.py",
    ]
    agent_proc = _spawn_process(agent_cmd, base_env, project_root, agent_log_path)
    processes.append(agent_proc)

    # Give agents time to initialize and register with the store.
    time.sleep(8)

    training_env = base_env.copy()
    training_env.setdefault("TRAINING_STRATEGY", "pure_mp")
    training_env.setdefault("MODEL_ID", "Qwen/Qwen3-0.6B")
    training_cmd = [
        "torchrun",
        "--nproc_per_node=4",
        "--master-port=39501",
        "prototyping/distributed_model_transfer/train.py",
    ]
    train_proc = _spawn_process(training_cmd, training_env, project_root, train_log_path)
    processes.append(train_proc)

    inference_env = base_env.copy()
    inference_env.setdefault("INFERENCE_STRATEGY", "hybrid_dp_mp")
    inference_env.setdefault("AGENT_RANK_OFFSET", "4")
    inference_env.setdefault("MODEL_ID", "Qwen/Qwen3-0.6B")
    inference_cmd = [
        "torchrun",
        "--nproc_per_node=4",
        "--master-port=39502",
        "prototyping/distributed_model_transfer/inference.py",
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
        shutil.rmtree(logs_dir, ignore_errors=True)
