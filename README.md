# Etha

Shard-to-shard weight transfer across device meshes.

Etha moves weights between any two `(DeviceMesh, placement)` layouts — typically
trainer to inference engine — without any rank ever materializing the full
tensor. Plans are computed once from placements and reused every sync.

## Design

Weight-sync mismatch between engines decomposes into four kinds — name, parallel,
affine layout, non-affine kernel format. Etha owns parallel (placement m2m);
affine fuse/transpose is encoded by the consumer's registered views; non-affine
(swizzle, quantization) stays in the engine. See
[docs/design/refactor-inprocess.md](docs/design/refactor-inprocess.md).

## Core

```
src/etha/
  ir.py            # Wire / Endpoint / Route / M2MMap / Chunk
  planner/
    m2m_map.py     # placement pair -> routes (lcm fingerprint, no payload)
    chunks.py      # routes + local tensors -> chunks (P2P / BROADCAST / LOCAL)
  execution.py     # chunk_comm: prepare -> wire ops -> finalize
  utils.py         # process-group cache, slice helpers
```

```python
m2m = get_m2m_map(src_mesh, src_placements, tgt_mesh, tgt_placements, group)
chunks = m2m_to_chunks(m2m, rank, source_tensor=src_shard, target_tensor=tar_shard)
chunk_comm(chunks)  # plan once, run every sync
```

Only dependency: torch.

## Develop

```
pixi run -e dev test     # CPU/gloo unit tests (linux-64 / osx-arm64)
pixi run -e gpu submit … # GPU runs on the cluster
```

Or with pip: `pip install -e . && pytest tests`.
