"""Profiler utilities."""

import os
import time
import pickle
import logging
import contextlib

import torch
from attr import dataclass
from upath import UPath

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Worker %(process)d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Memory snapshot configuration
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


@dataclass
class ProfilingSpec:
    enable_profiling: bool
    dump_folder: UPath
    save_traces_folder: UPath
    profile_freq: int
    warmup_steps: int
    active_steps: int
    enable_memory_snapshot: bool


@contextlib.contextmanager
def maybe_enable_profiling(profiling_spec: ProfilingSpec, *, global_step: int = 0):
    """Enable torch profiler.

    Args:
        profiling_spec: ProfilingSpec with profiling configuration
        global_step: Starting step number for profiler

    Yields:
        torch profiler or None if profiling disabled
    """
    if not profiling_spec.enable_profiling:
        yield contextlib.nullcontext()
        return

    # Setup directories
    trace_dir = profiling_spec.dump_folder / profiling_spec.save_traces_folder

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    # Create trace directory
    os.makedirs(trace_dir, exist_ok=True)

    def trace_handler(prof):
        """Handle profiler output."""
        curr_trace_dir_name = f"iteration_{prof.step_num}"
        curr_trace_dir = trace_dir / curr_trace_dir_name
        os.makedirs(curr_trace_dir, exist_ok=True)

        # Export Chrome trace
        trace_file = f"{curr_trace_dir}/rank{rank}_trace.pt.trace.json"
        logger.info(f"Dumping profiler traces at step {prof.step_num} to {trace_file}")

        begin = time.monotonic()
        prof.export_chrome_trace(trace_file)
        logger.info(f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds")

    logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

    warmup = profiling_spec.warmup_steps
    active = profiling_spec.active_steps
    profile_freq = profiling_spec.profile_freq
    wait = profile_freq - (active + warmup)
    assert wait >= 0, "profile_freq must be greater than or equal to warmup + active"

    # Setup profiler activities
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif torch.xpu.is_available():
        activities.append(torch.profiler.ProfilerActivity.XPU)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
        on_trace_ready=trace_handler,
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
    ) as torch_profiler:
        torch_profiler.step_num = global_step
        yield torch_profiler


def dump_memory_snapshot(output_dir: str, step: int, rank: int) -> None:
    """Dump a standalone memory snapshot without profiler.

    Args:
        output_dir: Directory to save the snapshot
        step: Current step number
        rank: Distributed rank
    """
    if not torch.cuda.is_available():
        return

    os.makedirs(output_dir, exist_ok=True)
    snapshot_file = f"{output_dir}/rank{rank}_memory_snapshot_step_{step}.pickle"

    try:
        begin = time.monotonic()
        with open(snapshot_file, "wb") as output:
            pickle.dump(torch.cuda.memory._snapshot(), output)
        logger.info(f"Memory snapshot saved: {snapshot_file} in {time.monotonic() - begin:.2f}s")
    except Exception as e:
        logger.warning(f"Failed to dump memory snapshot: {e}")
