"""Trainer side: build a grouped DTensor state_dict, then dcp.load.

Qwen3-30B-A3B specific. transformers `Qwen3MoeExperts` stores experts as
grouped 3D tensors that match vLLM's `w13_weight`/`w2_weight`. The
`GroupedMoEPlanner` re-keys the grouped entries to per-expert HF names
that DCP can find on disk; values are views into the grouped buffer.
"""

import os
import re
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from common import (
    CONFIG,
    MOE_HANDLERS,
    HANDLER_PLACEMENTS,
    get_handler_name,
    open_control_store,
    get_queue_state_paths,
)
from transformers import AutoConfig, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard, DTensor
from torch.distributed.checkpoint import DefaultLoadPlanner, HuggingFaceStorageReader

from etha.tensor_bus import TensorBusClient, bootstrap_client
from etha.tensor_bus.client import BatchHandler

logger = logging.getLogger(__name__)

_GROUPED_RE = re.compile(r"^(.+\.mlp\.experts)\.(gate_up_proj|down_proj)$")


def _explode_grouped_moe(state_dict: dict) -> dict:
    """Grouped MoE entries -> per-expert HF-keyed views into the grouped buffer."""
    out = {}
    for key, value in state_dict.items():
        m = _GROUPED_RE.match(key)
        if m is None:
            out[key] = value
            continue
        prefix, role = m.group(1), m.group(2)
        local = value._local_tensor if isinstance(value, DTensor) else value
        e_local = local.shape[0]
        # e_start: row-major chunk index across all mesh dims that shard tensor dim 0.
        chunk_idx = 0
        if isinstance(value, DTensor):
            mesh = value.device_mesh
            for mesh_dim, pl in enumerate(value.placements):
                if isinstance(pl, Shard) and pl.dim == 0:
                    rank = mesh.get_local_rank(mesh.mesh_dim_names[mesh_dim])
                    chunk_idx = chunk_idx * mesh.size(mesh_dim) + rank
        e_start = chunk_idx * e_local
        if role == "gate_up_proj":
            half = local.shape[1] // 2
            for i in range(e_local):
                out[f"{prefix}.{e_start + i}.gate_proj.weight"] = local[i, :half, :]
                out[f"{prefix}.{e_start + i}.up_proj.weight"] = local[i, half:, :]
        else:  # down_proj
            for i in range(e_local):
                out[f"{prefix}.{e_start + i}.down_proj.weight"] = local[i]
    return out


class GroupedMoEPlanner(DefaultLoadPlanner):
    def set_up_planner(self, state_dict, *args, **kwargs) -> None:
        super().set_up_planner(_explode_grouped_moe(state_dict), *args, **kwargs)


def _local_shape(global_shape: tuple[int, ...], mesh: DeviceMesh, placements: tuple) -> tuple[int, ...]:
    shape = list(global_shape)
    for dim_idx, pl in enumerate(placements):
        if isinstance(pl, Shard):
            shape[pl.dim] //= mesh.mesh.shape[dim_idx]
    return tuple(shape)


def _mesh_world_size(mesh: DeviceMesh) -> int:
    ws = 1
    for s in mesh.mesh.shape:
        ws *= int(s)
    return ws


def _setup_distributed() -> tuple[int, torch.device, DeviceMesh, DeviceMesh]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == CONFIG.trainer_world_size, (
        f"torchrun world_size={world_size} but CONFIG.trainer_world_size={CONFIG.trainer_world_size}"
    )
    attn_r = CONFIG.trainer_attn_dp_replicate
    attn_s = CONFIG.trainer_attn_dp_shard
    moe_r = CONFIG.trainer_moe_dp_replicate
    moe_s = CONFIG.trainer_moe_dp_shard
    ep = CONFIG.trainer_ep_size
    assert attn_r * attn_s == world_size, (
        f"attn_dp_replicate * attn_dp_shard = {attn_r * attn_s} != world_size={world_size}"
    )
    assert moe_r * moe_s * ep == world_size, (
        f"moe_dp_replicate * moe_dp_shard * ep = {moe_r * moe_s * ep} != world_size={world_size}"
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")

    att_mesh = DeviceMesh(
        "cuda",
        torch.arange(attn_r * attn_s).view(attn_r, attn_s),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    moe_mesh = DeviceMesh(
        "cuda",
        torch.arange(moe_r * moe_s * ep).view(moe_r, moe_s, ep),
        mesh_dim_names=("dp_replicate", "dp_shard", "ep"),
    )
    logger.info("rank=%d att_mesh=%s moe_mesh=%s", rank, att_mesh, moe_mesh)
    return rank, device, att_mesh, moe_mesh


def _build_state_dict(
    hf_config,
    att_mesh: DeviceMesh,
    moe_mesh: DeviceMesh,
    device: torch.device,
) -> dict[str, DTensor]:
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.float32, trust_remote_code=True)

    state: dict[str, DTensor] = {}
    for name, meta_param in model.named_parameters():
        handler = get_handler_name(name)
        if handler is None:
            continue
        placements = HANDLER_PLACEMENTS[handler]
        mesh = moe_mesh if handler in MOE_HANDLERS else att_mesh
        local = torch.empty(_local_shape(meta_param.shape, mesh, placements), dtype=meta_param.dtype, device=device)
        state[name] = DTensor.from_local(local, mesh, placements)
    return state


def _init_pairs(
    client: TensorBusClient,
    handlers: list[str],
    att_mesh: DeviceMesh,
    moe_mesh: DeviceMesh,
) -> None:
    for handler in handlers:
        placements = HANDLER_PLACEMENTS[handler]
        mesh = moe_mesh if handler in MOE_HANDLERS else att_mesh
        ws = _mesh_world_size(mesh)
        client.init_pair(
            pair_name=handler,
            local_name=CONFIG.trainer_name,
            remote_name=CONFIG.vllm_name,
            expected_world_size=ws,
            device_mesh=mesh,
            placements=placements,
            blocking=True,
            timeout=CONFIG.connection_timeout,
        )
        logger.info("init_pair %s ws=%d placements=%s", handler, ws, placements)


def _register_batch(
    client: TensorBusClient,
    state: dict[str, DTensor],
) -> tuple[BatchHandler, list[torch.Tensor]]:
    by_pair: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for name, dt in state.items():
        local = dt.to_local()
        # Boundary split: vLLM's CUTLASS MoE swaps w13 to [up; gate] post-load;
        # send per-projection views so vLLM side can place them in swapped slots.
        if name.endswith("mlp.experts.gate_up_proj"):
            half = local.shape[1] // 2
            base = name[: -len("gate_up_proj")]
            by_pair.setdefault("experts_gate_up", []).append((base + "gate_proj", local[:, :half, :]))
            by_pair["experts_gate_up"].append((base + "up_proj", local[:, half:, :]))
            continue
        by_pair.setdefault(get_handler_name(name), []).append((name, local))

    tensors_to_send: list[tuple[torch.Tensor, str]] = []
    local_store: list[torch.Tensor] = []
    for handler in sorted(by_pair.keys()):
        for _name, t in sorted(by_pair[handler], key=lambda x: x[0]):
            local_store.append(t)
            tensors_to_send.append((t, handler))
    batch = client.register_tensors(
        batch_id=CONFIG.batch_id,
        tensors=tensors_to_send,
        timeout=CONFIG.connection_timeout,
    )
    logger.info("registered %d tensors for batch %s", len(tensors_to_send), CONFIG.batch_id)
    return batch, local_store


def _wait_for_chats(control_store, baseline: int, target_delta: int, timeout: float = 3600.0) -> int:
    deadline = time.monotonic() + timeout
    target = baseline + target_delta
    while time.monotonic() < deadline:
        raw = control_store.get("chat_count")
        current = int(raw) if raw else 0
        if current >= target:
            return current
        time.sleep(CONFIG.not_ready_sleep)
    raise TimeoutError(f"chat_count did not reach {target} within {timeout}s")


def _sync_loop(
    batch: BatchHandler,
    local_store: list[torch.Tensor],
    rounds: int,
    interval: float,
    scale: float,
    control_store,
    rank: int,
) -> None:
    chat_baseline = 0
    if rank == 0 and control_store is not None:
        raw = control_store.get("chat_count")
        chat_baseline = int(raw) if raw else 0
        logger.info("starting chat_count baseline=%d", chat_baseline)

    for round_idx in range(rounds):
        if scale != 1.0:
            logger.info("round %d/%d perturbing weights (×%g)", round_idx + 1, rounds, scale)
            with torch.no_grad():
                for t in local_store:
                    t.mul_(scale)
        dist.barrier()
        logger.info("round %d/%d sending", round_idx + 1, rounds)
        batch.transfer(transfer_type="send", blocking=True, timeout=CONFIG.sync_timeout)
        logger.info("round %d/%d send complete", round_idx + 1, rounds)

        if rank == 0 and control_store is not None:
            target = chat_baseline + CONFIG.chats_per_round
            logger.info("round %d/%d waiting for chat_count >= %d", round_idx + 1, rounds, target)
            chat_baseline = _wait_for_chats(control_store, chat_baseline, CONFIG.chats_per_round)
            logger.info("round %d/%d chat_count reached %d", round_idx + 1, rounds, chat_baseline)
        dist.barrier()
        time.sleep(interval)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [trainer rk{os.environ.get('RANK')}] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    rank, device, att_mesh, moe_mesh = _setup_distributed()

    hf_config = AutoConfig.from_pretrained(CONFIG.model_id, trust_remote_code=True)
    if getattr(hf_config, "model_type", "") != "qwen3_moe":
        raise RuntimeError(f"Hard-coded for Qwen3-MoE, got model_type={hf_config.model_type!r}")

    model_dir = Path(snapshot_download(CONFIG.model_id))
    logger.info("checkpoint dir: %s", model_dir)

    state = _build_state_dict(hf_config, att_mesh, moe_mesh, device)
    logger.info("state_dict: %d tensors, dcp.load...", len(state))
    dcp.load(
        state,
        storage_reader=HuggingFaceStorageReader(str(model_dir)),
        planner=GroupedMoEPlanner(),
    )
    logger.info("load complete")

    client, info = bootstrap_client(path_naming_fn=get_queue_state_paths)
    logger.info("etha client bootstrapped, agent_rank=%d", info.agent_rank)

    handlers = sorted({get_handler_name(name) for name in state})
    _init_pairs(client, handlers, att_mesh, moe_mesh)
    batch, local_store = _register_batch(client, state)

    control_store = open_control_store() if rank == 0 else None
    if rank == 0:
        logger.info("control store connected on rank 0")

    try:
        _sync_loop(
            batch,
            local_store,
            CONFIG.sync_rounds,
            CONFIG.trainer_sync_interval,
            CONFIG.trainer_perturb_scale,
            control_store,
            rank,
        )
    finally:
        batch.close()
        client.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
