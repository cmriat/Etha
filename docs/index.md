# Etha

> M-to-N DTensor redistribute across PyTorch process groups — any (mesh, placement) → any (mesh, placement).

Etha redistributes a tensor described as `(DeviceMesh, Placement)` on one
PyTorch process group into a different `(DeviceMesh, Placement)` on a second,
independently-launched process group — the same redistribution `DTensor` does
in-process, generalized to two unrelated jobs.

The canonical use case: shipping model weights from a training cluster to an
inference cluster in a disaggregated RL setup, where the two sides were
launched separately and run different parallelism configurations.

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
