"""driver:唯一同时看到两边的进程(single-controller)。

起 vLLM(load_format="dummy",乱码权重)+ 连 trainer 的 collective RPC,
编排 init(声明互换、各端本地 build_chunks)与 sync(chunk_comm)。
验证:sync 前 generate 乱码,sync 后正常文本 = 权重传对。
"""

import os

from rpc import CollectiveClient
from vllm import LLM, SamplingParams
from trainer_server import RPC_PORT_BASE

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

    trainer = CollectiveClient([("127.0.0.1", RPC_PORT_BASE + i) for i in range(T)])
    manifest = trainer.collective_rpc("manifest")[0]
    t_decl = trainer.collective_rpc("etha_export")[0]
    v_decl = llm.collective_rpc("etha_export", args=(T, 1))[0]

    world = T + tp
    trainer.send_all("etha_init", "127.0.0.1", CROSS_PORT, world, list(manifest), v_decl)
    llm.collective_rpc("etha_init", args=("127.0.0.1", CROSS_PORT, world, manifest, t_decl))
    trainer.recv_all()

    trainer.send_all("etha_transfer")
    llm.collective_rpc("etha_transfer")
    trainer.recv_all()
    print("[after]", repr(llm.generate([prompt], sp)[0].outputs[0].text), flush=True)


if __name__ == "__main__":
    main()
