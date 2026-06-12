# Etha

Core-only rebuild in progress: planner (`m2m_map`/`chunks`) + `ir` + `execution`,
no engine integration yet. The spec is docs/design/refactor-inprocess.md — read it
before structural changes.

Run everything with `pixi run`. `dev` env works on macOS (CPU); GPU work goes
through `pixi run -e gpu submit <script>` (slurm via kjobctl).

Verify changes with `pixi run -e dev test` (gloo CPU reshard tests, fast).
NCCL paths: `pixi run -e gpu submit scripts/test_gpu.sbatch` from the coder.cpu
workspace (~/repos/etha-refactor; needs CONDA_OVERRIDE_CUDA=12.9 there, the
submit host has no GPU). Last verified green on 8xH20Z — cases + 20 fuzz rounds;
first cases are slow from NCCL lazy connection setup, steady-state is sub-ms.
