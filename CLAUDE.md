# Etha

Core-only rebuild in progress: planner (`m2m_map`/`chunks`) + `ir` + `execution`,
no engine integration yet. The spec is docs/design/refactor-inprocess.md — read it
before structural changes.

Run everything with `pixi run`. `dev` env works on macOS (CPU); GPU work goes
through `pixi run -e gpu submit <script>` (slurm via kjobctl).

Verify changes with `pixi run -e dev test` (gloo CPU reshard tests, fast).
GPU/NCCL paths have no local coverage — flag them for cluster verification.
