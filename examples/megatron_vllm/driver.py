"""driver:唯一同时看到两边的进程(single-controller)。

所有跨 world 元数据(清单、声明、store 地址)经 driver 搬运,全是 KB 级纯数据;
plan 在每端本地纯函数算出,worker 之间没有任何自协调。

worker 胶水(各端一个 RPC 入口,trainer 是 Ray actor 方法 / vLLM 是
worker_extension_cls 的方法),把 Protocol 实现包成三个 RPC:
  etha_export(manifest, topo, base_rank) -> shardings     构造 Protocol 实现,导出声明
  etha_init(host, port, world, peer_shardings)            create_cross_group + build_chunks 缓存
  etha_transfer()                                          chunk_comm;收端末尾 process_after_load
"""

import asyncio

from utils import free_port, ray_trainer, vllm_handle, group_moe_experts, load_safetensors_index  # pseudo


def main():
    model_path = "deepseek-ai/DeepSeek-V3"  # pseudo
    trainer = ray_trainer()  # pseudo: Ray actor 句柄(每 rank)
    vllm = vllm_handle()  # pseudo: AsyncLLM

    # ── rank 记账:trainer 占头段,replica 依次接续 ──────────────────────────
    T = trainer.world_size  # pseudo
    N = T + vllm.num_replicas * vllm.dp * vllm.tp  # pseudo

    # ── 清单:HF index 为权威(序即 canonical 序),一次性规约与裁剪 ──────────
    manifest = load_safetensors_index(model_path)  # {hf_name: (shape, dtype)}
    manifest = group_moe_experts(manifest)  # per-expert 名折叠成 grouped 名
    # 部署子集差异在此剔除(如 vLLM 未加载 MTP 头);此后两端查不到名字即 bug

    # ── init(一次):声明互换,plan 各端本地算 ─────────────────────────────
    port = free_port()
    t_decl = trainer.rpc("etha_export", manifest, base_rank=0)  # pseudo
    v_decl = vllm.collective_rpc("etha_export", manifest, base_rank=T)  # pseudo
    # 聚合:PP 各 stage 的声明拼合(per-param mesh 常态);tied embedding 同名多源任取一
    trainer.rpc("etha_init", "driver-host", port, N, peer=v_decl)  # pseudo
    vllm.collective_rpc("etha_init", "driver-host", port, N, peer=t_decl)  # pseudo

    # ── 每轮 sync ──────────────────────────────────────────────────────────
    async def sync():
        # 健康检查:RPC 失败即信号;成员变了 → abort 旧 cross PG → 重建 + plan 重算
        # (纯函数毫秒级)。窗口内故障由 NCCL watchdog + 短 timeout 兜底,见设计文档。
        await vllm.pause_generation()  # 真实 API,async_llm.py:723
        send = trainer.rpc_async("etha_transfer")  # pseudo: 两端同时进 chunk_comm
        vllm.collective_rpc("etha_transfer")  # UTILITY 在 step 边界执行
        send.wait()
        await vllm.resume_generation()

    asyncio.run(sync())


if __name__ == "__main__":
    main()
