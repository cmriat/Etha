# Etha

> Distributed P2P tensor transfer for PyTorch.
> Named after the [Sub-Etha](https://hitchhikers.fandom.com/wiki/Sub-Etha).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Etha is a tensor transfer library for PyTorch distributed jobs that need to
move tensors between two independently-launched process groups — for example,
shipping model weights from a training cluster to an inference cluster in a
disaggregated RL setup.

It plans communication on top of `DeviceMesh` + `Placement`, batches and
buckets small transfers, and uses NCCL for the actual send/recv.

## Architecture

```
                          ┌─────────────────┐
                          │     KVStore     │  (etcd or torch TCPStore)
                          │ rendezvous +    │
                          │ mesh exchange   │
                          └────────┬────────┘
                                   │
       Producer side               │           Consumer side
  ┌───────────────────────┐        │      ┌───────────────────────┐
  │ Worker  (user code)   │        │      │ Worker  (user code)   │
  │   └─ TensorBusClient  │        │      │   └─ TensorBusClient  │
  └──────────┬────────────┘        │      └──────────┬────────────┘
             │ LMDB CommandQueue   │                 │ LMDB CommandQueue
             ▼                     │                 ▼
  ┌───────────────────────┐        │      ┌───────────────────────┐
  │  Agent  (torchrun)    │◀───────┴──────▶│  Agent  (torchrun)   │
  │  NCCL process group   │   NCCL send/   │  NCCL process group  │
  │                       │   recv         │                      │
  └───────────────────────┘                └──────────────────────┘
```

- **Agent** processes own the NCCL process group and execute transfers. They
  are launched with `torchrun` and a single `world_size` that covers both
  sides.
- **Worker** processes (your training / inference code) use `TensorBusClient`
  to register tensors and issue send / recv. They never touch NCCL directly.
- **CommandQueue** (LMDB) is the worker → agent channel; commands carry a
  POSIX semaphore name so workers can block until the agent finishes.
- **KVStore** (etcd or torch TCPStore) handles rendezvous, namespace
  isolation, and exchange of mesh / placement metadata between the two sides.

Each `(local_name, remote_name, DeviceMesh, Placement)` tuple registers as a
**pair**. Tensors are registered into a **batch** that spans one or more pairs
and is then transferred atomically.

## Installation

Etha uses [pixi](https://pixi.sh/) for environment management.

```bash
git clone https://github.com/cmriat/Etha.git
cd Etha
pixi install -e dev
pixi shell -e dev
```

Requirements: Linux x86_64, CUDA 12.9, Python 3.12.

## Quick start

The minimal usage is symmetric on both sides — producer and consumer follow
the same shape.

```python
from etha.tensor_bus import TensorBusClient
from torch.distributed.tensor.placement_types import Shard

client = TensorBusClient(agent_rank=...)

client.init_pair(
    pair_name="weights",
    local_name="trainer",
    remote_name="inference",
    expected_world_size=4,
    device_mesh=mesh,
    placements=(Shard(0),),
)

handler = client.register_tensors(
    batch_id="step_0",
    tensors=[(t, "weights") for t in tensors],
)

handler.transfer("send", blocking=True)   # "recv" on the other side
handler.close()
```

A complete runnable example that transfers a Qwen3 model between two separate
`torchrun` groups lives in
[`prototyping/distributed_model_transfer/`](prototyping/distributed_model_transfer/).
Benchmarks are under [`bench/`](bench/).

## Repository layout

```
src/etha/
  comm/         M2M planning, chunking, bucketing, NCCL ops
  tensor_bus/   Agent / Client / CommandQueue / pair & batch state
  kvstore/      KVStore abstraction (etcd, torch TCPStore)
tests/          pytest suite
bench/          comm + KV store benchmarks
prototyping/    end-to-end runnable examples (not stable API)
docs/design/    design notes
```

## Development

```bash
pixi shell -e dev
pre-commit install
pytest tests
```

## License

Apache-2.0, see [LICENSE](LICENSE).
