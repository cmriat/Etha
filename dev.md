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

**M2 第一条端到端:torch-native → vLLM(disaggregated)✅ bf16 已通(真 server 模式)**
真实跑通(8×H20Z 单机):FSDP2 Qwen3-0.6B(4 卡)→ etha m2m → cross NCCL →
vLLM tp=4 shard 直落 → load_weights → dummy 乱码变正常生成。vLLM 走
`vllm serve` + 官方 /collective_rpc 端点(VLLM_SERVER_DEV_MODE);trainer 做成
vLLM 形状的 server(HTTP 入口 + zmq 扇出);driver 纯 HTTP 编排,不占 GPU、
不 import 引擎;跨边界元数据一律 base64(pickle) 信封(文本边界)。
上游 PR 范围修正:`is_sharded_weight` 只有 v1 weight_loader 检查,**v2
(parameter.py 的 load_*,bf16/主流量化都走它)没有旁路**——PR 要补 v2 +
embedding + MoE-EP-off;example 暂用"绑回 v1 loader"过渡。
✅ 已 engine 化:EthaWeightTransferEngine(官方插件位,纯传输零 model 依赖,
EthaInitInfo 即 init_info schema);extension 薄至 etha_export(= 未来 get_sharding);
官方 update_weights 自动包 layerwise(quant 管线免费,bf16 已实跑全管线)。
上游 PR 新增论据:backend Literal 封闭与 factory 注册制矛盾;trainer_send_weights
的 full-tensor 流假设不适配 shard-direct;layerwise record 快照会盖掉后打的
param attrs(需要打标后 re-record)。
✅ 两端对称:EthaInitInfo/EthaTrainerEngine 共享于 protocol.py(零 vllm 依赖,
duck-typed parse),trainer worker 用与 vLLM 同名的 RPC(init_weight_transfer_engine
/update_weights),driver 对两端发同形调用(仅 base_rank 与 self/peer 声明互换)。
✅ 流式内存:chunk_comm 的 target_alloc/on_complete 把 dst buffer 分配/释放纳入
窗口执行流(Chunk.weight 归属,m2m_to_chunks 接 target_shape 延迟落点)——峰值 =
在飞窗口的相邻权重 local shard + 最大单权重(硬下界),与层数无关;window 是旋钮。
实测(0.6B/tp4):in-flight buffer peak 0.316GB vs full-shard 0.531GB(降 40%),
残留 ≈ embedding 一块。
上游 PR 强论据(峰值显存):**峰值 floor = 最大单权重 local shard,而最大单权重恰是
embedding——它现在走 Replicate fallback(整块 0.31GB/rank)正因 VocabParallelEmbedding
的 loader 缺 is_sharded_weight 旁路**。补这个旁路不只是"对齐 Linear",还把 embedding
按 tp 切(0.31→0.078GB),峰值 floor 直接掉到下一个权重。所以 is_sharded_weight 补
v2+embedding+MoE-EP-off 是性能项(砍峰值),不只是正确性项。
(注:in-flight 计数器假设 on_complete 即释放;layerwise 对容器模块有 delayed
现象——结构性 numel 重复计数,no-op fallback,不影响正确性但会让真实 GPU 占用偏高。)
余项:量化档实测、多 replica、671B 基线、容错实测。

**M3 bench**
chain 链式广播带宽、窗口扫参、与旧 etha 对比(fanout 已删:chain 在权重同步的带宽域恒优)。

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
