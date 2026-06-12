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

import requests

from rpc import CollectiveClient

CROSS_PORT = 52701
VLLM_URL = "http://127.0.0.1:52300"


def enc(obj):
    return base64.b64encode(pickle.dumps(obj)).decode()


def dec(s):
    return pickle.loads(base64.b64decode(s))


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

    while True:
        try:
            if requests.get(f"{VLLM_URL}/health", timeout=5).status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(5)
    print("[before]", repr(generate(model, "The capital of France is")), flush=True)

    trainer = CollectiveClient()
    manifest = trainer.collective_rpc("manifest")[0]
    t_decl = trainer.collective_rpc("etha_export")[0]
    v_decl = dec(vllm_rpc("etha_export", T, 1)[0])

    world = T + tp
    wait = trainer.collective_rpc_async("etha_init", "127.0.0.1", CROSS_PORT, world, list(manifest), v_decl)
    vllm_rpc("etha_init", "127.0.0.1", CROSS_PORT, world, enc(manifest), enc(t_decl))
    wait()

    wait = trainer.collective_rpc_async("etha_transfer")
    vllm_rpc("etha_transfer")
    wait()
    print("[after]", repr(generate(model, "The capital of France is")), flush=True)


if __name__ == "__main__":
    main()
