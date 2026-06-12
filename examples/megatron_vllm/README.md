# Megatron → vLLM 权重传输

伪代码示例(megatron/vllm 的调用以 `# pseudo` 标注;etha 调用是真实 API)。
disaggregated 部署:训练 world 与 vLLM 推理实例各自已在运行,driver 是唯一
同时看到两边的进程。

核心约定见 `protocol.py`:**每个引擎实现一份 `EngineWeightProtocol`(三个方法),
driver 与 etha 不认识任何具体引擎**;权威清单 = HF checkpoint index(序即
canonical 序,单边独有的权重天然不在清单上,查不到名字即 bug)。

## 文件

| 文件 | 角色 |
|---|---|
| `protocol.py` | `EngineWeightProtocol`(get_sharding / local_view / process_after_load)+ `build_chunks`(每端本地的 plan 构建,route_idx 全局重编) |
| `trainer_side.py` | **Megatron 实现**:mesh 装配(parallel_state)+ placement 白名单 + 名字表 + qkv 去交错 view——命令式分片的声明化,四样手工 |
| `fsdp_side.py` | **torch-native 实现**(susser-tod/torchtitan 式):DTensor 直接读,三行——「免费」那格的实证 |
| `vllm_side.py` | **vLLM 收端**:统一 loader 路线——etha 只管并行,收端 buffer 是 HF 布局的本 rank shard,摆放交给引擎自己的 `load_weights`(`is_sharded_weight` 跳 narrow);quant 配置包进 layerwise reload |
| `driver.py` | 编排:rank 记账、清单规约、声明互换、每轮 pause → transfer → resume |

## 时序

```
INIT(一次)
  driver:  清单 = HF index(MoE 折叠 grouped 名,按部署裁剪)
  两端:    etha_export(manifest, base_rank) → {hf_name: (mesh, placements)}  ← KB 级
  driver:  聚合(PP stage 拼合、tied 去重),互换声明
  两端:    etha_init —— create_cross_group + build_chunks(纯函数,本地)缓存

每轮 sync
  driver:  健康检查(成员变 → abort + 重建 + plan 重算)
  driver:  vllm.pause_generation()                         ← 真实 API
  并行:    trainer chunk_comm(发)  ←→  vllm chunk_comm(收)
  vllm:    process_after_load(清单) —— load_weights 摆放(+ quant 档 layerwise process)
  driver:  vllm.resume_generation()
```

## 运行(伪)

```bash
pixi run -e megatron torchrun ... trainer_side.py     # 训练侧(已有作业内嵌)
pixi run -e vllm ... vllm_side.py                     # 推理侧(已有 server 内嵌)
python driver.py                                      # 编排
```
