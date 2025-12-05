#!/usr/bin/env python3
"""Benchmark comparing EtcdStore vs TorchTCPStore performance.

Usage:
    pixi run -e dev python bench/kvstore_benchmark.py
    pixi run -e dev python bench/kvstore_benchmark.py --num-ops 1000
    pixi run -e dev python bench/kvstore_benchmark.py --backend etcd
    pixi run -e dev python bench/kvstore_benchmark.py --help
"""

import os
import time
import statistics
from typing import Literal
from dataclasses import dataclass

import tyro

# Set environment variables before any imports that use network
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")


@dataclass
class Config:
    """KVStore benchmark configuration."""

    # etcd server host
    etcd_host: str = "localhost"
    # etcd server port
    etcd_port: int = 2379
    # TCPStore server host
    tcp_host: str = "localhost"
    # TCPStore server port
    tcp_port: int = 29501
    # Number of operations for set/get/exists benchmarks
    num_ops: int = 500
    # Which backend to benchmark
    backend: Literal["both", "etcd", "tcp"] = "both"


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    operation: str
    count: int
    total_time_ms: float
    avg_time_us: float
    std_time_us: float
    ops_per_sec: float


def benchmark_set(store, num_ops: int, key_prefix: str) -> BenchmarkResult:
    """Benchmark set operations."""
    times = []
    for i in range(num_ops):
        key = f"{key_prefix}/key:{i}"
        value = f"value_{i}"
        start = time.perf_counter()
        store.set(key, value)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1e6)

    total_ms = sum(times) / 1000
    return BenchmarkResult(
        name=store.__class__.__name__,
        operation="set",
        count=num_ops,
        total_time_ms=total_ms,
        avg_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        ops_per_sec=num_ops / (total_ms / 1000),
    )


def benchmark_get(store, num_ops: int, key_prefix: str) -> BenchmarkResult:
    """Benchmark get operations (keys must exist)."""
    for i in range(num_ops):
        store.set(f"{key_prefix}/key:{i}", f"value_{i}")

    times = []
    for i in range(num_ops):
        key = f"{key_prefix}/key:{i}"
        start = time.perf_counter()
        _ = store.get(key)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1e6)

    total_ms = sum(times) / 1000
    return BenchmarkResult(
        name=store.__class__.__name__,
        operation="get",
        count=num_ops,
        total_time_ms=total_ms,
        avg_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        ops_per_sec=num_ops / (total_ms / 1000),
    )


def benchmark_exists(store, num_ops: int, key_prefix: str) -> BenchmarkResult:
    """Benchmark exists operations."""
    for i in range(num_ops):
        store.set(f"{key_prefix}/key:{i}", f"value_{i}")

    times = []
    for i in range(num_ops):
        key = f"{key_prefix}/key:{i}"
        start = time.perf_counter()
        _ = store.exists(key)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1e6)

    total_ms = sum(times) / 1000
    return BenchmarkResult(
        name=store.__class__.__name__,
        operation="exists",
        count=num_ops,
        total_time_ms=total_ms,
        avg_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        ops_per_sec=num_ops / (total_ms / 1000),
    )


def benchmark_set_bytes(store, num_ops: int, key_prefix: str, data_size: int) -> BenchmarkResult:
    """Benchmark set_bytes operations."""
    data = b"x" * data_size
    times = []
    for i in range(num_ops):
        key = f"{key_prefix}/key:{i}"
        start = time.perf_counter()
        store.set_bytes(key, data)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1e6)

    total_ms = sum(times) / 1000
    return BenchmarkResult(
        name=store.__class__.__name__,
        operation=f"set_bytes({data_size}B)",
        count=num_ops,
        total_time_ms=total_ms,
        avg_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        ops_per_sec=num_ops / (total_ms / 1000),
    )


def benchmark_get_bytes(store, num_ops: int, key_prefix: str, data_size: int) -> BenchmarkResult:
    """Benchmark get_bytes operations (keys must exist)."""
    data = b"x" * data_size
    for i in range(num_ops):
        store.set_bytes(f"{key_prefix}/key:{i}", data)

    times = []
    for i in range(num_ops):
        key = f"{key_prefix}/key:{i}"
        start = time.perf_counter()
        _ = store.get_bytes(key)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1e6)

    total_ms = sum(times) / 1000
    return BenchmarkResult(
        name=store.__class__.__name__,
        operation=f"get_bytes({data_size}B)",
        count=num_ops,
        total_time_ms=total_ms,
        avg_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        ops_per_sec=num_ops / (total_ms / 1000),
    )


def benchmark_wait_for_key_existing(store, num_ops: int, key_prefix: str) -> BenchmarkResult:
    """Benchmark wait_for_key when key already exists."""
    for i in range(num_ops):
        store.set(f"{key_prefix}/key:{i}", f"value_{i}")

    times = []
    for i in range(num_ops):
        key = f"{key_prefix}/key:{i}"
        start = time.perf_counter()
        _ = store.wait_for_key(key, timeout=10.0)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1e6)

    total_ms = sum(times) / 1000
    return BenchmarkResult(
        name=store.__class__.__name__,
        operation="wait_for_key(existing)",
        count=num_ops,
        total_time_ms=total_ms,
        avg_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        ops_per_sec=num_ops / (total_ms / 1000),
    )


def benchmark_wait_for_key_async(store, num_ops: int, key_prefix: str, delay_ms: float = 10) -> BenchmarkResult:
    """Benchmark wait_for_key when key is written asynchronously."""
    import threading

    times = []
    for i in range(num_ops):
        key = f"{key_prefix}/async:{i}"

        def writer(k=key):
            time.sleep(delay_ms / 1000)
            store.set(k, "async_value")

        thread = threading.Thread(target=writer)
        thread.start()

        start = time.perf_counter()
        _ = store.wait_for_key(key, timeout=10.0)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1e6)

        thread.join()

    total_ms = sum(times) / 1000
    return BenchmarkResult(
        name=store.__class__.__name__,
        operation=f"wait_for_key(async,{delay_ms}ms)",
        count=num_ops,
        total_time_ms=total_ms,
        avg_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        ops_per_sec=num_ops / (total_ms / 1000),
    )


def benchmark_wait_for_keys_etcd(store, num_keys: int, key_prefix: str) -> BenchmarkResult:
    """Benchmark wait_for_keys for EtcdStore (keys already exist)."""
    for i in range(num_keys):
        store.set(f"{key_prefix}/rank:{i}/ready", "1")

    pattern = f"{key_prefix}/rank:*/ready"
    num_iterations = 10

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = store.wait_for_keys(pattern, expected_count=num_keys, timeout=10.0)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1e6)

    total_ms = sum(times) / 1000
    return BenchmarkResult(
        name=store.__class__.__name__,
        operation=f"wait_for_keys({num_keys})",
        count=num_iterations,
        total_time_ms=total_ms,
        avg_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        ops_per_sec=num_iterations / (total_ms / 1000),
    )


def benchmark_wait_for_keys_tcp(store, num_keys: int, key_prefix: str) -> BenchmarkResult:
    """Benchmark wait_for_keys for TorchTCPStore (keys already exist)."""
    candidate_keys = []
    for i in range(num_keys):
        key = f"{key_prefix}/rank:{i}/ready"
        store.set(key, "1")
        candidate_keys.append(key)

    pattern = f"{key_prefix}/rank:*/ready"
    num_iterations = 10

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = store.wait_for_keys(
            pattern,
            expected_count=num_keys,
            timeout=10.0,
            candidate_keys=candidate_keys,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1e6)

    total_ms = sum(times) / 1000
    return BenchmarkResult(
        name=store.__class__.__name__,
        operation=f"wait_for_keys({num_keys})",
        count=num_iterations,
        total_time_ms=total_ms,
        avg_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        ops_per_sec=num_iterations / (total_ms / 1000),
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a table."""
    print("\n" + "=" * 80)
    print(f"{'Store':<20} {'Operation':<25} {'Avg (μs)':<12} {'Std (μs)':<12} {'Ops/sec':<12}")
    print("=" * 80)

    for r in results:
        print(f"{r.name:<20} {r.operation:<25} {r.avg_time_us:<12.2f} {r.std_time_us:<12.2f} {r.ops_per_sec:<12.0f}")

    print("=" * 80)


def run_etcd_benchmark(cfg: Config) -> list[BenchmarkResult]:
    """Run benchmarks for EtcdStore."""
    from etha.kvstore import EtcdStore

    print(f"\n[EtcdStore] Connecting to {cfg.etcd_host}:{cfg.etcd_port}...")
    store = EtcdStore(host=cfg.etcd_host, port=cfg.etcd_port)

    results = []
    prefix = f"bench/etcd/{time.time()}"

    print("[EtcdStore] Running set benchmark...")
    results.append(benchmark_set(store, cfg.num_ops, f"{prefix}/set"))

    print("[EtcdStore] Running get benchmark...")
    results.append(benchmark_get(store, cfg.num_ops, f"{prefix}/get"))

    print("[EtcdStore] Running exists benchmark...")
    results.append(benchmark_exists(store, cfg.num_ops, f"{prefix}/exists"))

    print("[EtcdStore] Running set_bytes benchmark...")
    for data_size in [64, 1024, 4096]:
        results.append(benchmark_set_bytes(store, cfg.num_ops, f"{prefix}/set_bytes/{data_size}", data_size))

    print("[EtcdStore] Running get_bytes benchmark...")
    for data_size in [64, 1024, 4096]:
        results.append(benchmark_get_bytes(store, cfg.num_ops, f"{prefix}/get_bytes/{data_size}", data_size))

    print("[EtcdStore] Running wait_for_key benchmark...")
    results.append(benchmark_wait_for_key_existing(store, cfg.num_ops, f"{prefix}/wait_key"))
    results.append(benchmark_wait_for_key_async(store, min(cfg.num_ops, 50), f"{prefix}/wait_key_async", delay_ms=10))

    print("[EtcdStore] Running wait_for_keys benchmark...")
    for num_keys in [8, 16, 32, 64]:
        results.append(benchmark_wait_for_keys_etcd(store, num_keys, f"{prefix}/wait/{num_keys}"))

    store.close()
    return results


def run_tcp_benchmark(cfg: Config) -> list[BenchmarkResult]:
    """Run benchmarks for TorchTCPStore."""
    from etha.kvstore import TorchTCPStore

    print(f"\n[TorchTCPStore] Starting server at {cfg.tcp_host}:{cfg.tcp_port}...")
    store = TorchTCPStore(
        host=cfg.tcp_host,
        port=cfg.tcp_port,
        world_size=1,
        is_master=True,
        timeout=60.0,
        wait_for_workers=False,
    )

    results = []
    prefix = f"bench/tcp/{time.time()}"

    print("[TorchTCPStore] Running set benchmark...")
    results.append(benchmark_set(store, cfg.num_ops, f"{prefix}/set"))

    print("[TorchTCPStore] Running get benchmark...")
    results.append(benchmark_get(store, cfg.num_ops, f"{prefix}/get"))

    print("[TorchTCPStore] Running exists benchmark...")
    results.append(benchmark_exists(store, cfg.num_ops, f"{prefix}/exists"))

    print("[TorchTCPStore] Running set_bytes benchmark...")
    for data_size in [64, 1024, 4096]:
        results.append(benchmark_set_bytes(store, cfg.num_ops, f"{prefix}/set_bytes/{data_size}", data_size))

    print("[TorchTCPStore] Running get_bytes benchmark...")
    for data_size in [64, 1024, 4096]:
        results.append(benchmark_get_bytes(store, cfg.num_ops, f"{prefix}/get_bytes/{data_size}", data_size))

    print("[TorchTCPStore] Running wait_for_key benchmark...")
    results.append(benchmark_wait_for_key_existing(store, cfg.num_ops, f"{prefix}/wait_key"))
    results.append(benchmark_wait_for_key_async(store, min(cfg.num_ops, 50), f"{prefix}/wait_key_async", delay_ms=10))

    print("[TorchTCPStore] Running wait_for_keys benchmark...")
    for num_keys in [8, 16, 32, 64]:
        results.append(benchmark_wait_for_keys_tcp(store, num_keys, f"{prefix}/wait/{num_keys}"))

    store.close()
    return results


def print_comparison(all_results: list[BenchmarkResult]) -> None:
    """Print comparison summary between etcd and TCPStore."""
    etcd_results = {r.operation: r for r in all_results if r.name == "EtcdStore"}
    tcp_results = {r.operation: r for r in all_results if r.name == "TorchTCPStore"}

    if not (etcd_results and tcp_results):
        return

    print("\n" + "=" * 80)
    print("Comparison (EtcdStore vs TorchTCPStore):")
    print("=" * 80)
    for op in etcd_results:
        if op in tcp_results:
            etcd_avg = etcd_results[op].avg_time_us
            tcp_avg = tcp_results[op].avg_time_us
            ratio = tcp_avg / etcd_avg if etcd_avg > 0 else float("inf")
            faster = "etcd" if etcd_avg < tcp_avg else "TCPStore"
            print(f"{op:<25}: {faster} is {abs(ratio - 1) * 100:.1f}% faster")


def main(cfg: Config) -> None:
    """Run KVStore benchmark."""
    all_results = []

    if cfg.backend in ("both", "etcd"):
        try:
            all_results.extend(run_etcd_benchmark(cfg))
        except Exception as e:
            print(f"\n[EtcdStore] Failed to connect: {e}")

    if cfg.backend in ("both", "tcp"):
        try:
            all_results.extend(run_tcp_benchmark(cfg))
        except Exception as e:
            print(f"\n[TorchTCPStore] Failed: {e}")

    if all_results:
        print_results(all_results)
        print_comparison(all_results)


if __name__ == "__main__":
    main(tyro.cli(Config))
