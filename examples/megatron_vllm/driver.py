"""driver:纯编排进程,不占 GPU,两边都是已在跑的 server。

vLLM 走真 server 模式(vllm serve + VLLM_SERVER_DEV_MODE 的 /collective_rpc
端点);trainer 是 vLLM 形状的自建 server(HTTP 入口 + zmq 扇出)。跨边界
元数据一律 base64(pickle) 信封——/collective_rpc 的约定就是只传字符串,
方法自己负责反序列化。两边的 collective(cross group 会合 / chunk_comm)
必须同时在飞:asyncio.gather。验证:dummy 乱码 → sync → 正常文本。
"""

import asyncio
import base64
import os
import pickle

import httpx
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from rpc import HTTP_PORT

CROSS_PORT = HTTP_PORT + 200
CROSS_HOST = os.environ.get("ETHA_CROSS_HOST", "127.0.0.1")  # 多节点:头节点 IP(cross-group rendezvous)
VLLM_URL = f"http://127.0.0.1:{os.environ.get('ETHA_VLLM_PORT', 52300)}"  # driver 与 vLLM 同节点
TRAINER_URL = f"http://{os.environ.get('ETHA_TRAINER_HOST', '127.0.0.1')}:{HTTP_PORT}"  # 多节点:trainer 在头节点


def enc(obj):
    return base64.b64encode(pickle.dumps(obj)).decode()


def dec(s):
    return pickle.loads(base64.b64decode(s))


async def trainer(client, method, *args):
    body = pickle.dumps((args, {})) if args else b""
    r = await client.post(f"{TRAINER_URL}/{method}", content=body)
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return pickle.loads(r.content)


async def vllm(client, method, *args):
    r = await client.post(f"{VLLM_URL}/collective_rpc", json={"method": method, "args": list(args)})
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return r.json()["results"] if r.content else None


async def generate(client, model, prompt):
    r = await client.post(
        f"{VLLM_URL}/v1/completions",
        json={"model": model, "prompt": prompt, "temperature": 0, "max_tokens": 24},
    )
    return r.json()["choices"][0]["text"]


async def wait_ready(request):
    while True:
        try:
            await request()
            return
        except httpx.ConnectError:
            await asyncio.sleep(5)


def build_manifest(model):
    """权威清单 = transformers 的模型定义(meta-init,零分配,秒级)。

    第三方、框架无关:transformers 定义 canonical 架构(MoE 融合名
    experts.gate_up_proj、dense 分开的 q/k/v),两端都往它映射(transformers
    trainer identity、Megatron converter、vLLM w13→gate_up_proj)。比 safetensors
    强在它是运行时逻辑权重(融合),不是可能 per-expert 的存储格式;比从 trainer
    拿强在不绑具体训练框架。
    """
    cfg = AutoConfig.from_pretrained(model)
    with torch.device("meta"):
        ref = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    manifest = {n: tuple(p.shape) for n, p in ref.named_parameters()}  # 只要几何;dtype 从源拿
    if getattr(cfg, "tie_word_embeddings", False):
        manifest.pop("lm_head.weight", None)  # tied:运行时两端都不持有独立副本
    return manifest


async def main():
    model = os.environ.get("ETHA_MODEL", "Qwen/Qwen3-0.6B")
    T = int(os.environ.get("TRAINER_WORLD", "4"))
    tp = int(os.environ.get("VLLM_TP", "4"))
    manifest = build_manifest(model)

    async with httpx.AsyncClient(timeout=3600) as client:
        await asyncio.gather(
            wait_ready(lambda: client.get(f"{VLLM_URL}/health")),
            wait_ready(lambda: client.post(f"{TRAINER_URL}/ping")),
        )
        print("[before]", repr(await generate(client, model, "The capital of France is")), flush=True)

        t_decl_b64 = (await trainer(client, "etha_export", 0))[0]
        v_decl_b64 = (await vllm(client, "etha_export", T))[0]

        # 两端完全同形:EthaInitInfo 即 init_info schema,self/peer 声明互换
        common = {"host": CROSS_HOST, "port": CROSS_PORT, "world": T + tp, "manifest": enc(manifest)}
        await asyncio.gather(
            trainer(client, "init_weight_transfer_engine",
                    {**common, "base_rank": 0, "self_decl": t_decl_b64, "peer_decl": v_decl_b64}),
            vllm(client, "init_weight_transfer_engine",
                 {**common, "base_rank": T, "self_decl": v_decl_b64, "peer_decl": t_decl_b64}),
        )
        # 多轮 sync:RL 每步都同步,验证 plan 缓存复用 + layerwise 重入 + re-record 跨轮持久
        for r in range(2):
            await asyncio.gather(
                trainer(client, "update_weights", {}),
                vllm(client, "update_weights", {}),
            )
            out = await generate(client, model, "The capital of France is")
            print(f"[after round {r}]", repr(out), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
