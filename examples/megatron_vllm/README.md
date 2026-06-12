# Megatron → vLLM 权重传输

伪代码示例(megatron/vllm 的调用以 `# pseudo` 标注;etha 调用是真实 API)。
disaggregated 部署:Megatron 训练 world 与 vLLM 推理实例各自已在运行,
driver 是唯一同时看到两边的进程。

## 拓扑

```
Megatron world(PP×DP×TP,rank 0..T-1)      vLLM replicas(R 个实例 × dp×tp,rank T..N-1)
        │  trainer_side.py(每 rank)                │  vllm_side.py(每 worker,经 collective_rpc)
        └────────────────┬───────────────────────────┘
                         │ driver.py(rank 记账 / 元数据收集 / plan / 每轮 sync 触发)
                  TCPStore + create_cross_group(NCCL,init 一次)
```

## 文件

| 文件 | 跑在哪 | 干什么 |
|---|---|---|
| `driver.py` | driver 进程 | rank 记账;收集两端声明;按 name join + 去重 + `route_idx` 重编;每轮 sync 编排(pause → transfer → process → resume) |
| `trainer_side.py` | 每个 Megatron rank | name converter(Megatron 名 → HF 名)+ placement converter(Megatron 分片 → mesh/placements)+ 每轮 `chunk_comm`(发) |
| `vllm_side.py` | 每个 vLLM worker | per-param 声明(placement 映射表 + fuse view spec)+ view 注册(直落/staging 两档)+ 每轮 `chunk_comm`(收)+ 触发 `process_weights_after_loading` |

## 时序

```
INIT(一次)
  driver:   分配 cross-world rank 段;开 TCPStore
  两端:     create_cross_group(host, port, my_cross_rank, N)
  trainer:  上报 {hf_name: (mesh_tensor, placements)}        ← converter 产出,KB 级
  vllm:     上报 {hf_name: (mesh_tensor, placements)} + view 注册在本地完成
  driver:   join by hf_name(同名多源去重)→ 下发每端的 per-param 清单(name 序)
  两端:     get_m2m_map(纯函数,本地)+ m2m_to_chunks + route_idx 重编 → chunks 缓存

每轮 sync
  driver:   vllm.pause()                                      # pseudo
  并行:     trainer 端 chunk_comm(chunks, group)   ←→   vllm 端 chunk_comm(chunks, group)
  vllm:     需要 process 的层:process_weights_after_loading   # pseudo
  driver:   vllm.resume()                                     # pseudo
```

## 运行(伪)

```bash
pixi run -e megatron torchrun ... trainer_side.py     # 训练侧(已有作业内嵌)
pixi run -e vllm ... vllm_side.py                     # 推理侧(已有 server 内嵌)
python driver.py                                      # 编排
```
