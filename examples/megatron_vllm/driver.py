"""driver:纯编排进程,不占 GPU,两边都是已在跑的 server。

vLLM 走真 server 模式(vllm serve + VLLM_SERVER_DEV_MODE 的 /collective_rpc
端点);trainer 是 vLLM 形状的自建 server(HTTP 入口 + zmq 扇出)。跨边界
元数据一律 base64(pickle) 信封——/collective_rpc 的约定就是只传字符串,
方法自己负责反序列化。验证:dummy 乱码 → sync → 正常文本。
"""

import base64
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from huggingface_hub import get_safetensors_metadata

from rpc import HTTP_PORT

CROSS_PORT = 52701
VLLM_URL = "http://127.0.0.1:52300"
TRAINER_URL = f"http://127.0.0.1:{HTTP_PORT}"
_pool = ThreadPoolExecutor(2)


def enc(obj):
    return base64.b64encode(pickle.dumps(obj)).decode()


def dec(s):
    return pickle.loads(base64.b64decode(s))


def post(url, *args):
    body = pickle.dumps((args, {})) if args else b""
    r = requests.post(url, data=body, timeout=3600)
    outs = []
    for status, val in pickle.loads(r.content):
        if status == "err":
            raise RuntimeError(val)
        outs.append(val)
    return outs


def concurrently(*calls):
    """两边的 collective(cross group 会合 / chunk_comm)必须同时在飞,全部等齐。"""
    futs = [_pool.submit(fn, *args) for fn, *args in calls]
    return [f.result() for f in futs]


def wait_ready(url, payload=None):
    while True:
        try:
            if payload is not None:
                requests.post(url, data=payload, timeout=5)
            elif requests.get(url, timeout=5).status_code != 200:
                raise requests.exceptions.ConnectionError
            return
        except requests.exceptions.ConnectionError:
            time.sleep(5)


def vllm_rpc(method, *args):
    r = requests.post(f"{VLLM_URL}/collective_rpc", json={"method": method, "args": list(args)}, timeout=3600)
    r.raise_for_status()
    return r.json()["results"] if r.content else None


def generate(model, prompt):
    r = requests.post(
        f"{VLLM_URL}/v1/completions",
        json={"model": model, "prompt": prompt, "temperature": 0, "max_tokens": 24},
        timeout=600,
    )
    return r.json()["choices"][0]["text"]


def main():
    model = os.environ.get("ETHA_MODEL", "Qwen/Qwen3-0.6B")
    T = int(os.environ.get("TRAINER_WORLD", "4"))
    tp = int(os.environ.get("VLLM_TP", "4"))

    # 权威清单:HF checkpoint index(第三方),不从任何一端拿;dtype 按 safetensors 命名
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    from transformers import AutoConfig

    meta = get_safetensors_metadata(model)
    manifest = {
        name: (tuple(info.shape), dtypes[info.dtype])
        for fm in meta.files_metadata.values()
        for name, info in fm.tensors.items()
    }
    if getattr(AutoConfig.from_pretrained(model), "tie_word_embeddings", False):
        manifest.pop("lm_head.weight", None)   # tied:checkpoint 的冗余副本,运行时两端都不持有

    wait_ready(f"{VLLM_URL}/health")
    wait_ready(f"{TRAINER_URL}/ping", b"")
    print("[before]", repr(generate(model, "The capital of France is")), flush=True)

    t_decl = post(f"{TRAINER_URL}/etha_export")[0]
    v_decl = dec(vllm_rpc("etha_export", T, 1)[0])

    world = T + tp
    concurrently(
        (post, f"{TRAINER_URL}/etha_init", "127.0.0.1", CROSS_PORT, world, list(manifest), v_decl),
        (vllm_rpc, "etha_init", "127.0.0.1", CROSS_PORT, world, enc(manifest), enc(t_decl)),
    )
    concurrently(
        (post, f"{TRAINER_URL}/etha_transfer"),
        (vllm_rpc, "etha_transfer"),
    )
    print("[after]", repr(generate(model, "The capital of France is")), flush=True)


if __name__ == "__main__":
    main()
