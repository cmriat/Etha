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
from huggingface_hub import get_safetensors_metadata
from transformers import AutoConfig

from rpc import HTTP_PORT

CROSS_PORT = HTTP_PORT + 200
VLLM_URL = f"http://127.0.0.1:{os.environ.get('ETHA_VLLM_PORT', 52300)}"
TRAINER_URL = f"http://127.0.0.1:{HTTP_PORT}"


def enc(obj):
    return base64.b64encode(pickle.dumps(obj)).decode()


def dec(s):
    return pickle.loads(base64.b64decode(s))


async def trainer(client, method, *args):
    body = pickle.dumps((args, {})) if args else b""
    r = await client.post(f"{TRAINER_URL}/{method}", content=body)
    r.raise_for_status()
    return pickle.loads(r.content)


async def vllm(client, method, *args):
    r = await client.post(f"{VLLM_URL}/collective_rpc", json={"method": method, "args": list(args)})
    r.raise_for_status()
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
    """权威清单:HF checkpoint index(第三方),不从任何一端拿。"""
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    meta = get_safetensors_metadata(model)
    manifest = {
        name: (tuple(info.shape), dtypes[info.dtype])
        for fm in meta.files_metadata.values()
        for name, info in fm.tensors.items()
    }
    if getattr(AutoConfig.from_pretrained(model), "tie_word_embeddings", False):
        manifest.pop("lm_head.weight", None)  # tied:checkpoint 的冗余副本,运行时两端都不持有
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

        t_decl = (await trainer(client, "etha_export", 0))[0]
        v_decl_b64 = (await vllm(client, "etha_export", T))[0]

        # init 走 worker 官方入口;EthaInitInfo 即 init_info 的 schema,
        # engine 零 model 依赖,自声明由 driver 回灌
        init_info = {
            "host": "127.0.0.1",
            "port": CROSS_PORT,
            "world": T + tp,
            "base_rank": T,
            "manifest": enc(manifest),
            "self_decl": v_decl_b64,
            "peer_decl": enc(t_decl),
        }
        await asyncio.gather(
            trainer(client, "etha_init", "127.0.0.1", CROSS_PORT, T + tp, list(manifest), dec(v_decl_b64)),
            vllm(client, "init_weight_transfer_engine", init_info),
        )
        await asyncio.gather(
            trainer(client, "etha_transfer"),
            vllm(client, "update_weights", {}),
        )
        print("[after]", repr(await generate(client, model, "The capital of France is")), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
