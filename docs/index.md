# Etha

> M-to-N DTensor redistribute across PyTorch process groups — any (mesh, placement) → any (mesh, placement).

Etha redistributes a tensor described as `(DeviceMesh, Placement)` on one
PyTorch process group into a different `(DeviceMesh, Placement)` on a second,
independently-launched process group — the same redistribution `DTensor` does
in-process, generalized to two unrelated jobs.

The canonical use case: shipping model weights from a training cluster to an
inference cluster in a disaggregated RL setup, where the two sides were
launched separately and run different parallelism configurations.

:::{note}
Supported source placements: `Shard`, `Replicate`, `Partial`. For `Partial`,
Etha collapses it to `Replicate` on the source mesh via a chunk-level
all-reduce before send. `Partial` on the target side is rejected:
cross-process-group decomposition of a logical tensor into a Partial
contribution is not uniquely defined.
:::

```{toctree}
:caption: Design
:maxdepth: 2

design/index
```

```{toctree}
:caption: API Reference
:maxdepth: 2

api/etha/index
```
