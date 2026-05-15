"""vLLM OpenAI server + worker-side weight-sync hooks (one file, two roles).

API-server patches `init_app_state` to wire `setup_tensorbus` and
`_sync_loop` after the engine is ready. `_sync_loop` polls cheap LMDB
signals and only pauses generation + dispatches `receive_weights` when the
trainer is actually sending. Worker-side hooks are cloudpickled and run
via `collective_rpc`; placements come from walking the loaded vLLM model.
"""

import os
import sys
import asyncio
import logging
from enum import Enum
from typing import Any
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import uvloop
from common import (
    CONFIG,
    get_handler_name,
    open_control_store,
    get_queue_state_paths,
    convert_vllm_state_dict,
)
from vllm.logger import init_logger
from sortedcontainers import SortedDict
from vllm.distributed import get_ep_group, get_world_group
from torch.distributed import DeviceMesh
from vllm.entrypoints.openai import api_server
from torch.distributed.tensor import Shard, Replicate
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.worker_base import WorkerBase
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.model_executor.layers.linear import RowParallelLinear, ColumnParallelLinear
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from etha.tensor_bus import bootstrap_client

logger = init_logger(__name__)


# --------------------------------------------------------------------------
# Worker side — runs inside each vLLM worker subprocess via collective_rpc.
# --------------------------------------------------------------------------


class MeshKind(Enum):
    DPTP = "dptp"
    EP = "ep"


def _get_ep_info(model: torch.nn.Module) -> dict[str, int] | None:
    for _, module in model.named_modules():
        if isinstance(module, FusedMoE):
            return {"ep_rank": module.ep_rank, "ep_size": module.ep_size}
    return None


def _build_device_meshes(self: WorkerBase) -> None:
    config = self.parallel_config
    dp = config.data_parallel_size
    tp = config.tensor_parallel_size
    self.dptp_device_mesh = DeviceMesh(
        mesh_dim_names=["dp", "tp"],
        mesh=torch.arange(dp * tp).view(dp, tp),
        device_type="cuda",
        _init_backend=False,
    )

    ep_info = _get_ep_info(self.get_model())
    if ep_info is not None:
        self.ep_device_mesh = DeviceMesh(
            mesh_dim_names=["ep"],
            mesh=torch.arange(ep_info["ep_size"]).view(ep_info["ep_size"]),
            device_type="cuda",
            _init_backend=False,
        )
    else:
        self.ep_device_mesh = None


def _get_placements(self: WorkerBase) -> dict[str, tuple[MeshKind, Any, tuple]]:
    """Map param_name -> (mesh_kind, device_mesh, placements) by walking vLLM modules."""
    if not hasattr(self, "dptp_device_mesh"):
        _build_device_meshes(self)

    placements: dict[str, tuple[MeshKind, Any, tuple]] = {}
    for module_name, module in self.get_model().named_modules():
        match module:
            case ColumnParallelLinear():
                kind, mesh = MeshKind.DPTP, self.dptp_device_mesh
                weight_pl = (Replicate(), Shard(0))
                bias_pl = (Replicate(), Shard(0))
            case RowParallelLinear():
                kind, mesh = MeshKind.DPTP, self.dptp_device_mesh
                weight_pl = (Replicate(), Shard(1))
                bias_pl = (Replicate(), Replicate())
            case VocabParallelEmbedding():
                kind, mesh = MeshKind.DPTP, self.dptp_device_mesh
                weight_pl = (Replicate(), Shard(0))
                bias_pl = None
            case FusedMoE():
                kind, mesh = MeshKind.EP, self.ep_device_mesh
                weight_pl = (Shard(0),)
                bias_pl = (Shard(0),)
            case _:
                kind, mesh = MeshKind.DPTP, self.dptp_device_mesh
                weight_pl = (Replicate(), Replicate())
                bias_pl = (Replicate(), Replicate())

        for para_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{para_name}"
            if "weight" in para_name and weight_pl is not None:
                placements[full_name] = (kind, mesh, weight_pl)
            elif "bias" in para_name and bias_pl is not None:
                placements[full_name] = (kind, mesh, bias_pl)
            elif para_name == "sinks":
                # GPT-OSS attention sinks: TP-sharded along head axis.
                placements[full_name] = (MeshKind.DPTP, self.dptp_device_mesh, (Replicate(), Shard(0)))

    return placements


def setup_tensorbus(self: WorkerBase) -> dict[str, Any]:
    local_rank = get_world_group().rank

    os.environ["RANK"] = str(local_rank)
    os.environ["AGENT_RANK_OFFSET"] = str(CONFIG.trainer_world_size)

    if not hasattr(self, "dptp_device_mesh"):
        _build_device_meshes(self)

    client, _ = bootstrap_client(
        path_naming_fn=get_queue_state_paths,
        connection_timeout=CONFIG.connection_timeout,
    )
    self._etha_client = client

    placements = _get_placements(self)
    converted = convert_vllm_state_dict(self.get_model().state_dict(), self.model_config.hf_config)

    # FusedMoE backends that swap w13 `[gate; up]` → `[up; gate]` in
    # `process_weights_after_loading` (FLASHINFER_CUTLASS / TRTLLM). All other
    # backends (TRITON, BATCHED_TRITON, AITER, ...) keep the HF layout.
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend

    experts_mod = self.get_model().model.layers[0].mlp.experts
    backend = getattr(experts_mod.quant_method, "unquantized_backend", None)
    self._etha_cutlass_swap = backend in (
        UnquantizedMoeBackend.FLASHINFER_CUTLASS,
        UnquantizedMoeBackend.FLASHINFER_TRTLLM,
    )
    logger.info("MoE backend=%s cutlass_swap=%s", backend, self._etha_cutlass_swap)

    config = self.parallel_config
    dptp_world_size = config.data_parallel_size * config.tensor_parallel_size
    ep_group = get_ep_group()
    ep_world_size = ep_group.world_size if ep_group is not None else 1

    pair_cfgs = SortedDict()
    tensors_by_pair: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)

    for param_name, (org_param_name, tensor) in converted.items():
        pair_name = get_handler_name(param_name)
        if pair_name is None:
            # No handler -> not transferred (e.g. KV-cache scales `_k_scale`, `_v_scale`).
            continue
        if org_param_name not in placements:
            continue
        kind, mesh, placement = placements[org_param_name]
        if pair_name not in pair_cfgs:
            pair_cfgs[pair_name] = dict(
                device_mesh=mesh,
                world_size=dptp_world_size if kind == MeshKind.DPTP else ep_world_size,
                placements=placement,
            )
        if param_name.endswith("mlp.experts.gate_up_proj"):
            # Boundary split: vLLM stores `w13` as `[gate; up]` under TRITON
            # and as `[up; gate]` under CUTLASS (post-load swap). Detect by
            # comparing the in-memory first-half against the HF gate weight.
            half = tensor.shape[1] // 2
            base = param_name[: -len("gate_up_proj")]
            cutlass_swap = getattr(self, "_etha_cutlass_swap", False)
            if cutlass_swap:
                gate_view, up_view = tensor[:, half:, :], tensor[:, :half, :]
            else:
                gate_view, up_view = tensor[:, :half, :], tensor[:, half:, :]
            tensors_by_pair[pair_name].append((base + "gate_proj", gate_view))
            tensors_by_pair[pair_name].append((base + "up_proj", up_view))
            continue
        tensors_by_pair[pair_name].append((param_name, tensor))

    for pair_name, cfg in pair_cfgs.items():
        client.init_pair(
            pair_name=pair_name,
            local_name=CONFIG.vllm_name,
            remote_name=CONFIG.trainer_name,
            expected_world_size=cfg["world_size"],
            device_mesh=cfg["device_mesh"],
            placements=cfg["placements"],
            blocking=True,
            timeout=CONFIG.connection_timeout,
        )

    all_tensors = []
    for pair_name in sorted(pair_cfgs.keys()):
        for tensor_name, tensor in sorted(tensors_by_pair[pair_name], key=lambda x: x[0]):  # noqa: B007
            all_tensors.append((tensor, pair_name))

    self._etha_batch_handler = client.register_tensors(
        batch_id=CONFIG.batch_id,
        tensors=all_tensors,
        timeout=CONFIG.connection_timeout,
    )

    logger.info("Etha initialized on rank=%d: %d tensors, %d pairs", local_rank, len(all_tensors), len(pair_cfgs))
    return {"status": "initialized", "rank": local_rank, "tensor_count": len(all_tensors)}


def query_transfer_signal(self: WorkerBase, timeout: float = 5.0) -> bool:
    if not hasattr(self, "_etha_batch_handler"):
        return False
    try:
        return self._etha_batch_handler.query_transfer_signal(blocking=True, timeout=timeout)
    except TimeoutError:
        return False


def receive_weights(self: WorkerBase, timeout: float = 120.0) -> dict[str, Any]:
    if not hasattr(self, "_etha_batch_handler"):
        raise RuntimeError("Etha not initialized — call setup_tensorbus first")

    local_rank = get_world_group().rank

    # Fence CUDA graphs around the weight write to avoid read/write races.
    torch.cuda.synchronize()
    logger.info("Receiving weights on rank=%d", local_rank)
    self._etha_batch_handler.transfer(transfer_type="recv", blocking=True, timeout=timeout)
    torch.cuda.synchronize()
    logger.info("Weights received on rank=%d", local_rank)

    return {"status": "received", "rank": local_rank}


# --------------------------------------------------------------------------
# API server side — runs in the main vllm_server.py Python process.
# --------------------------------------------------------------------------


def _build_args():
    return make_arg_parser(FlexibleArgumentParser()).parse_args(
        [
            "--model",
            CONFIG.model_id,
            "--tensor-parallel-size",
            str(CONFIG.vllm_tp_size),
            "--data-parallel-size",
            str(CONFIG.vllm_dp_size),
            "--enable-expert-parallel",
            "--dtype",
            "bfloat16",
            "--enforce-eager",
            "--load-format",
            "dummy",
            "--trust-remote-code",
            "--disable-log-stats",
            "--host",
            CONFIG.vllm_http_host,
            "--port",
            str(CONFIG.vllm_http_port),
        ]
    )


def _patch_init_app_state() -> None:
    original = api_server.init_app_state

    async def patched(engine_client, state, args, *extra_args, **extra_kwargs):
        await original(engine_client, state, args, *extra_args, **extra_kwargs)
        logger.info("setting up TensorBus on workers")
        await engine_client.collective_rpc(setup_tensorbus)
        logger.info("TensorBus ready, starting sync loop")
        state.weight_sync_task = asyncio.create_task(_sync_loop(engine_client))

    api_server.init_app_state = patched


async def _sync_loop(engine_client) -> None:
    store = open_control_store()
    store.set("vllm_ready", "1")
    store.set("vllm_version", "0")
    version = 0
    logger.info("control store ready (vllm_ready=1, vllm_version=0)")

    while True:
        try:
            signals = await engine_client.collective_rpc(query_transfer_signal, args=(CONFIG.sync_query_timeout,))
            if not any(signals):
                await asyncio.sleep(CONFIG.sync_poll_interval)
                continue

            store.set("vllm_ready", "0")
            logger.info("transfer_signal seen, entering recv (version %d -> %d)", version, version + 1)
            # clear_cache: stale KV under old weights would garble decode under new ones.
            await engine_client.pause_generation(mode="abort", clear_cache=True)
            try:
                await engine_client.collective_rpc(receive_weights, args=(CONFIG.sync_timeout,))
            finally:
                await engine_client.resume_generation()
            version += 1
            store.set("vllm_version", str(version))
            store.set("vllm_ready", "1")
            logger.info("sync round %d complete (vllm_ready=1)", version)
        except asyncio.CancelledError:
            logger.info("sync loop cancelled")
            raise
        except Exception:
            logger.exception("sync round failed, retrying in 5s")
            store.set("vllm_ready", "1")
            await asyncio.sleep(5)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [vllm-server] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _patch_init_app_state()
    uvloop.run(api_server.run_server(_build_args()))


if __name__ == "__main__":
    main()
