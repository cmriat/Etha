"""Profiler utilities."""

import os
import time
import logging
import contextlib

import torch
from attr import dataclass
from upath import UPath

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Inference Worker %(process)d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


@dataclass
class ProfilingSpec:
    dump_folder: UPath
    save_traces_folder: UPath
    upath: UPath
    profile_freq: int
    warmup_steps: int
    active_steps: int


@contextlib.contextmanager
def maybe_enable_profiling(profiling_spec, *, global_step: int = 0):
    # get user defined profiler settings
    enable_profiling = profiling_spec.profiling.enable_profiling

    if enable_profiling:
        dump_dir = profiling_spec.dump_folder
        save_trace_dir = profiling_spec.save_traces_folder
        trace_dir = dump_dir / save_trace_dir
        profile_freq = profiling_spec.profile_freq
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = trace_dir / curr_trace_dir_name
            if not curr_trace_dir.exists():
                os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()
            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.pt.trace.json")
            logger.info(f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds")

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not trace_dir.exists():
            os.makedirs(trace_dir, exist_ok=True)

        warmup, active = profiling_spec.warmup_steps, profiling_spec.active_steps
        wait = profile_freq - (active + warmup)
        assert wait >= 0, "profile_freq must be greater than or equal to warmup + active"
        gpu_device_profiled = None
        if torch.cuda.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.CUDA
        elif torch.xpu.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.XPU
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                gpu_device_profiled,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            with_stack=False,
            record_shapes=False,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


@contextlib.contextmanager
def maybe_enable_profiling(profiling_spec: ProfilingSpec, *, global_step: int = 0):
    # get user defined profiler settings
    enable_profiling = profiling_spec.enable_profiling

    if enable_profiling:
        dump_dir = profiling_spec.dump_folder
        save_trace_dir = profiling_spec.save_traces_folder
        trace_dir = dump_dir / save_trace_dir
        profile_freq = profiling_spec.profile_freq
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = trace_dir / curr_trace_dir_name
            if not curr_trace_dir.exists():
                os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()
            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.pt.trace.json")
            logger.info(f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds")

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not trace_dir.exists():
            os.makedirs(trace_dir, exist_ok=True)

        warmup, active = profiling_spec.warmup_steps, profiling_spec.active_steps
        wait = profile_freq - (active + warmup)
        assert wait >= 0, "profile_freq must be greater than or equal to warmup + active"
        gpu_device_profiled = None
        if torch.cuda.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.CUDA
        elif torch.xpu.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.XPU
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                gpu_device_profiled,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            with_stack=False,
            record_shapes=False,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None
