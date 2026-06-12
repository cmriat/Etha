"""driver:唯一同时看到两边的进程(single-controller)。

起 vLLM(load_format="dummy",乱码权重)+ 连 trainer 的 collective RPC,
编排 init(声明互换、各端本地 build_chunks)与 sync(chunk_comm)。
验证:sync 前 generate 乱码,sync 后正常文本 = 权重传对。
"""

import os

import torch

from rpc import CollectiveClient
from vllm import LLM, SamplingParams

CROSS_PORT = 52701


def main():
    model = os.environ.get("ETHA_MODEL", "Qwen/Qwen3-0.6B")
    T = int(os.environ.get("TRAINER_WORLD", "4"))
    tp = int(os.environ.get("VLLM_TP", "4"))

    llm = LLM(
        model=model,
        load_format="dummy",
        tensor_parallel_size=tp,
        enforce_eager=True,
        gpu_memory_utilization=0.6,
        worker_extension_cls="vllm_side.EthaWorkerExtension",
    )
    sp = SamplingParams(temperature=0, max_tokens=24)
    prompt = "The capital of France is"
    print("[before]", repr(llm.generate([prompt], sp)[0].outputs[0].text), flush=True)

    trainer = CollectiveClient()
    manifest = trainer.collective_rpc("manifest")[0]
    t_decl = trainer.collective_rpc("etha_export")[0]
    v_decl = llm.collective_rpc("etha_export", args=(T, 1))[0]
    # vLLM 的 msgpack 把 tensor/tuple 还原成嵌套 list,转回声明形态再转发
    v_decl = {n: (torch.as_tensor(mesh), tuple(pl)) for n, (mesh, pl) in v_decl.items()}

    world = T + tp
    wait = trainer.collective_rpc_async("etha_init", "127.0.0.1", CROSS_PORT, world, list(manifest), v_decl)
    llm.collective_rpc("etha_init", args=("127.0.0.1", CROSS_PORT, world, manifest, t_decl))
    wait()

    wait = trainer.collective_rpc_async("etha_transfer")
    llm.collective_rpc("etha_transfer")
    wait()
    print("[after]", repr(llm.generate([prompt], sp)[0].outputs[0].text), flush=True)


if __name__ == "__main__":
    main()
