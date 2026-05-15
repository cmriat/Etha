"""Shared config + Qwen3-MoE state-dict converter.

Converter aligns names on both sides:
    qkv_proj.weight       -> q_proj / k_proj / v_proj views
    experts.w13_weight    -> experts.gate_up_proj  (E, 2I, H)
    experts.w2_weight     -> experts.down_proj    (E, H, I)
"""

import time
import logging
from types import SimpleNamespace
from collections import OrderedDict

import torch
from torch.distributed.tensor import Shard, Placement, Replicate

from etha.kvstore.tcp import TorchTCPStore

logger = logging.getLogger(__name__)

CONFIG = SimpleNamespace(
    trainer_world_size=4,
    vllm_world_size=4,
    vllm_dp_size=2,
    vllm_tp_size=2,
    trainer_attn_dp_replicate=2,
    trainer_attn_dp_shard=2,
    trainer_moe_dp_replicate=1,
    trainer_moe_dp_shard=2,
    trainer_ep_size=2,
    model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
    batch_id="weight_sync",
    trainer_name="trainer",
    vllm_name="vllm",
    store_host="localhost",
    store_port=49001,
    store_backend="tcp",
    # Separate namespace from tensor_bus's own keys on the same store.
    control_namespace="sync_example",
    control_component="control",
    lmdb_root="/tmp/dbs_weight_sync",
    connection_timeout=300.0,
    sync_timeout=300.0,
    sync_poll_interval=0.3,
    sync_query_timeout=1.0,
    sync_rounds=3,
    trainer_sync_interval=1.0,
    trainer_perturb_scale=1.0,
    chats_per_round=3,
    vllm_http_host="localhost",
    vllm_http_port=8000,
    chat_interval=1.0,
    chat_rounds=0,
    not_ready_sleep=0.2,
)
CONFIG.agent_world_size = CONFIG.trainer_world_size + CONFIG.vllm_world_size


def open_control_store(timeout: float = 60.0) -> TorchTCPStore:
    """Connect as a client to the agent rank-0 TCPStore, retrying until reachable."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return TorchTCPStore(
                host=CONFIG.store_host,
                port=CONFIG.store_port,
                world_size=1,
                is_master=False,
                wait_for_workers=False,
                timeout=30.0,
                namespace=CONFIG.control_namespace,
                component=CONFIG.control_component,
            )
        except Exception as e:
            last_err = e
            logger.info("control store not reachable yet (%s), retrying", e)
            time.sleep(2)
    raise TimeoutError(
        f"control store at {CONFIG.store_host}:{CONFIG.store_port} not reachable after {timeout}s: {last_err}"
    )


def get_queue_state_paths(rank: int) -> tuple[str, str]:
    return (f"{CONFIG.lmdb_root}/{rank}_command.lmdb", f"{CONFIG.lmdb_root}/{rank}_state.lmdb")


HANDLER_KEYWORDS: tuple[tuple[str, str], ...] = (
    # Order matters: most-specific keyword first.
    ("q_norm", "layernorm"),
    ("k_norm", "layernorm"),
    ("input_layernorm", "layernorm"),
    ("post_attention_layernorm", "layernorm"),
    ("model.norm", "layernorm"),
    ("embed_tokens", "embed_tokens"),
    ("q_proj", "qkv_proj"),
    ("k_proj", "qkv_proj"),
    ("v_proj", "qkv_proj"),
    ("o_proj", "o_proj"),
    ("mlp.gate.weight", "router"),
    ("experts.gate_up_proj", "experts_gate_up"),
    ("experts.down_proj", "experts_down"),
    ("lm_head", "lm_head"),
)

MOE_HANDLERS = frozenset({"experts_gate_up", "experts_down"})

# Trainer-side placements. Hierarchy follows susser-tod's mesh naming:
#   att_mesh = (dp_replicate, dp_shard) 2D, all Replicate — attn/dense
#     weights fully replicated across all trainer ranks.
#   moe_mesh = (dp_replicate, dp_shard, ep) 3D: dp_shard and ep both
#     shard the E axis (dim 0). DCP load writes plain-tensor per-expert
#     views (one per HF safetensors key), so each view must be the full
#     HF per-expert shape `(I, H)` / `(H, I)`. Sharding only the E axis
#     keeps each view intact while distributing experts across the
#     dp_shard × ep grid.
HANDLER_PLACEMENTS: dict[str, tuple[Placement, ...]] = {
    "embed_tokens": (Replicate(), Shard(0)),
    "qkv_proj": (Replicate(), Shard(0)),
    "o_proj": (Replicate(), Shard(1)),
    "router": (Replicate(), Replicate()),
    "experts_gate_up": (Replicate(), Shard(0), Shard(0)),
    "experts_down": (Replicate(), Shard(0), Shard(0)),
    "lm_head": (Replicate(), Shard(0)),
    "layernorm": (Replicate(), Replicate()),
}


def get_handler_name(param_name: str) -> str | None:
    for kw, handler in HANDLER_KEYWORDS:
        if kw in param_name:
            return handler
    return None


def _qkv_split(qkv: torch.Tensor, hf_config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head_dim = (
        hf_config.head_dim if hasattr(hf_config, "head_dim") else hf_config.hidden_size // hf_config.num_attention_heads
    )
    num_q_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    total_heads = num_q_heads + 2 * num_kv_heads

    qkv_3d = qkv.view(-1, head_dim, qkv.shape[-1])
    scale = total_heads / qkv_3d.shape[0]
    nq = int(num_q_heads / scale)
    nkv = int(num_kv_heads / scale)
    q = qkv_3d[:nq].reshape(-1, qkv.shape[-1])
    k = qkv_3d[nq : nq + nkv].reshape(-1, qkv.shape[-1])
    v = qkv_3d[nq + nkv : nq + 2 * nkv].reshape(-1, qkv.shape[-1])
    return q, k, v


def convert_vllm_state_dict(
    state_dict: dict[str, torch.Tensor], hf_config
) -> "OrderedDict[str, tuple[str, torch.Tensor]]":
    """vLLM state dict -> {trainer_name: (vllm_orig_name, tensor)}."""  # noqa: D403
    out: OrderedDict[str, tuple[str, torch.Tensor]] = OrderedDict()
    for vllm_name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        if vllm_name.endswith("self_attn.qkv_proj.weight"):
            base = vllm_name[: -len("self_attn.qkv_proj.weight")]
            q, k, v = _qkv_split(tensor, hf_config)
            out[base + "self_attn.q_proj.weight"] = (vllm_name, q)
            out[base + "self_attn.k_proj.weight"] = (vllm_name, k)
            out[base + "self_attn.v_proj.weight"] = (vllm_name, v)
            continue

        # FusedMoE w13/w2 already match transformers Qwen3MoeExperts grouped layout.
        if vllm_name.endswith("mlp.experts.w13_weight"):
            new_name = vllm_name[: -len("w13_weight")] + "gate_up_proj"
            out[new_name] = (vllm_name, tensor)
            continue
        if vllm_name.endswith("mlp.experts.w2_weight"):
            new_name = vllm_name[: -len("w2_weight")] + "down_proj"
            out[new_name] = (vllm_name, tensor)
            continue

        out[vllm_name] = (vllm_name, tensor)
    return out
