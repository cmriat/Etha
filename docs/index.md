# Etha

> M-to-N tensor transfer between PyTorch process groups.

Etha is a tensor transfer library for PyTorch distributed jobs that need to
move tensors between two independently-launched process groups — for example,
shipping model weights from a training cluster to an inference cluster in a
disaggregated RL setup.

It plans the cross-process-group **resharding** (`DeviceMesh` + `Placement`
on each side) once per pair, then specializes that plan per batch into
NCCL send / recv ops bucketed for throughput.

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
