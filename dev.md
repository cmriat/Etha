# 跨引擎权重传输标准化

megatron、torch-native 框架(torchtitan / VeOmni / Automodel)、vllm、sglang 等之间
权重传输标准化。

Etha = 跨 world 的 redistribute。差异只来自四类:name / parallel / 仿射 / 非仿射;
每个引擎提供一份到 universal 中间表示的映射(N+M,不是 N×M),Etha 在中间做
never-full 的 placement m2m。详见 docs/design/refactor-inprocess.md。

## 引擎矩阵:每个引擎欠什么

| 引擎 | name(↔HF 清单) | parallel(placement) | 仿射 | 非仿射 |
|---|---|---|---|---|
| torch-native(源) | 贴 HF,近零 | **免费**:读 `param.placements`(含 `_StridedShard`)——fsdp_side.py 三行 | `to_local()` | 无(发送侧不存在) |
| Megatron(源) | 名字表 + PP 全局层号(trainer_side.py) | placement 白名单 converter(trainer_side.py 雏形,M4 完善) | qkv 去交错等 = 发送 view | 无 |
| vLLM(收) | `packed_modules_mapping` 等引擎自带 | placement rule 表(vllm_side.py);长期上游 `get_sharding` | **统一 loader 路线**:`is_sharded_weight` 跳切分,摆放由 `load_weights` 真跑;quant 包 layerwise reload | loader 管线内自动 |
| SGLang(收) | 待调研对应物 | 待调研 | 待调研(loader 结构类似则同路线) | 待调研 |

## 里程碑

**M0 核心 ✅**(已完成)
纯函数 planner(标记重捕)/ 链式流水 execution / cross-world bootstrap /
strided 支持 / 随机对拍 fuzz。torch-only,CPU 测试 macOS 可跑。

**M1 IPC(colocate 路径)**
`Chunk` 的 IPC 形态(LOCAL + 映射 src)+ handle export/import 工具 + in-place 约束。
解锁:同卡跨进程交付;torch-native colocate = 组内 m2m + IPC 落点(verl 的
shard-to-shard 替代 all_gather full tensor)。

**M2 第一条端到端:torch-native → vLLM(disaggregated)**
设计已在 examples/megatron_vllm 伪代码定型(Protocol 三方法 + 清单权威 +
统一 loader 路线 + build_chunks),待真实化:worker RPC 胶水、
`is_sharded_weight` 上游 PR(embedding / MoE-EP-off 对齐 Linear 语义)。
集群 GPU 验证:对齐 671B ~1s 基线;容错实测(kill 一个 replica →
abort 旧 cross PG → 重建 → 下轮 sync 正常)。

**M3 bench**
chain vs fanout A/B(`split_fanout` 开关)、窗口扫参、与旧 etha 对比。

**M4 Megatron 源**
Megatron-sharding → placement converter + name/仿射归一化(EP×ETP 用
`(Replicate, Shard(0), Shard(1))` 表达,核心已覆盖)。

**M5 SGLang 收端 + 上游化**
SGLang 四类对应物调研;vLLM `get_sharding`/`get_layout` RFC(版本漂移的根治,
#36222 类 bug 即论据)。

**M6 transport 扩展**(触发条件驱动,设计已存档)
NIXL/Mooncake:传输计算重叠或弹性扩缩容需求出现时再做;pull 方向 +
注册策略 + pool 轮翻转的结论在设计文档 transport 节。

## 量化策略(定论)

零依赖的两个形态进设计:原生 fp8 训练直发 fp8+scale(伴生 tensor,1× 带宽);
bf16 训练直发 bf16,量化留收端 `process_weights_after_loading`(2× 带宽换
trainer 零依赖)。同 recipe 的发端 cast(带宽减半)是可选优化,代价是 trainer
侧 quant 依赖与 recipe 对拍维护——同步带宽成为实测瓶颈时再启用。
