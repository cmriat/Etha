# 跨引擎权重传输标准化

megatron、torch-native 框架(torchtitan / VeOmni / Automodel)、vllm、sglang 等之间
权重传输标准化。

Etha = 跨 world 的 redistribute。差异只来自四类:name / parallel / 仿射 / 非仿射;
每个引擎提供一份到 universal 中间表示的映射(N+M,不是 N×M),Etha 在中间做
never-full 的 placement m2m。详见 docs/design/refactor-inprocess.md。

## 引擎矩阵:每个引擎欠什么

| 引擎 | name(↔HF 名) | parallel(placement) | 仿射(布局 view) | 非仿射 |
|---|---|---|---|---|
| torch-native(源) | 框架命名 ↔ HF,薄 converter | **免费**:读 `param.placements`(含 `_StridedShard`) | 通常无(贴 HF 布局) | 无(发送侧不存在) |
| Megatron(源) | `linear_qkv` 等 ↔ HF,converter(verl 有现成可参考) | **缺**:命令式分布 → placement converter 待写 | interleaved qkv 等归一化 | 无 |
| vLLM(收) | `WeightsMapper` + `packed_modules_mapping`(现成) | 短期 driver 手表装配(见设计文档映射表);长期上游 `get_sharding` | `get_layout`(fuse view 注册),短期手表 | `process_weights_after_loading`(现成,需可单独触发) |
| SGLang(收) | 待调研对应物 | 同上,待调研 | 待调研 | 待调研 |

## 里程碑

**M0 核心 ✅**(已完成)
纯函数 planner(标记重捕)/ 链式流水 execution / cross-world bootstrap /
strided 支持 / 随机对拍 fuzz。torch-only,CPU 测试 macOS 可跑。

**M1 IPC(colocate 路径)**
`Chunk` 的 IPC 形态(LOCAL + 映射 src)+ handle export/import 工具 + in-place 约束。
解锁:同卡跨进程交付;torch-native colocate = 组内 m2m + IPC 落点(verl 的
shard-to-shard 替代 all_gather full tensor)。

**M2 第一条端到端:torch-native → vLLM(disaggregated)**
driver 装配层:rank 记账 / 统一 mesh / placement 表(设计文档 driver 节落代码);
name join(HF 名主键,同名多源去重)+ `route_idx` 全局重编 helper;consumer 侧
view 注册——**直落为默认**(bf16→bf16 / fp8→fp8,process no-op,零 staging),
process 真干活的配置走裸层循环 staging。trainer PP 自动覆盖(per-param mesh)。
集群 GPU 验证,对齐 671B ~1s 基线。

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
