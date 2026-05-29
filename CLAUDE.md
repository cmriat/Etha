# Etha

Run everything with `pixi run`. `dev` and `vllm` both define a `submit` task, so the
env flag is mandatory: `pixi run -e dev <task>`.

## Verifying changes to `src/etha/comm/` or `src/etha/tensor_bus/`

Unit tests only cover disjoint-mesh CPU paths — run all three layers:

1. Unit tests (CPU/gloo, ~9 min):
   ```
   pixi run -e dev pytest tests/test_communication_cpu.py \
     tests/test_communication_symmetric_mesh.py \
     tests/test_communication_replicate_shard.py tests/test_partial_chunk_reduce.py
   ```
2. Transfer benchmark (GPU/NCCL): `pixi run -e dev submit bench/run_benchmark.sh`.
   Quick single-node check (P2P/BROADCAST/SHADOW paths):
   ```
   ETHA_BENCH_SMOKE=1 pixi run -e dev torchrun --nnodes=1 --nproc-per-node=8 \
     --master-addr=localhost --master-port=29555 ./bench/transfer_benchmark.py
   ```
3. vLLM weight-sync example (8 GPUs, the only end-to-end production-path check):
   `pixi run -e vllm vllm_weight_sync`. Set `HF_HOME=/data/hf` (cached 30B model).
   Long-running — monitor under an agent, not inline.

Gotchas:
- vLLM env build failing with `Permission denied` in `.pixi/build/work/`: a stale
  read-only git checkout — `chmod -R u+w .pixi/build/work/vllm-*` then retry.
