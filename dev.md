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
✅ **MoE 已通**(Qwen3-30B-A3B,EP on,8×H20Z):FSDP2 → expert 维融合 reshard →
feed-slice → vLLM EP,dummy 乱码变连贯文本。关键设计:
- **manifest = transformers reference**(meta-init named_parameters):第三方、框架无关
  的 canonical 架构(融合 MoE 名 experts.gate_up_proj、分开 q/k/v);只取 shape;
- **dtype 从源声明拿**(get_sharding 返回 (mesh, placements, dtype)),量化/混合精度
  buffer 才对;
- **MoE = expert 维(dim0)融合 reshard**:transformers 5.x 和 vLLM expert 张量布局
  完全相同 → (R, S(0), S(0));收端把本地融合 buffer **feed-slice 成 per-expert** 喂
  native loader(global expert id = ep_rank*local+i);
- 踩坑:untied lm_head 是顶层无点模块(30B tie_word_embeddings=False 才暴露,rpartition)。
版本调查结论:**native vLLM 模型仍是 per-expert 加载**(fused_moe_make_expert_params_mapping),
只有 Transformers backend(非主流)支持直接喂融合 experts.gate_up_proj——故 feed-slice。
✅ 多轮 sync 验证:RL 每步同步,round 0/1 都正常——plan 缓存复用、layerwise 重入、re-record 跨轮持久。
✅ **DeepSeek 架构已通**(DeepSeek-V2-Lite,MLA + shared experts + DeepSeek MoE,EP on):
**零新代码首跑过**——MLA 的 q_a/kv_a(ReplicatedLinear→Replicate)、kv_b/o_proj(Column/Row)、
shared experts(dense MLP 走 dense 路)、routed experts(SharedFusedMoE 是 FusedMoE 子类→_moe_shardings)、
first_k_dense_replace(layer0 dense)全被现有规则 + transformers reference manifest 覆盖。
**框架通用性验证**:同一份代码三种架构全通(Qwen3 dense / Qwen3 MoE / DeepSeek MLA+MoE),
无 per-model 硬编码——这是相对 Aaron 硬编码 placement 表的核心优势。
671B(DeepSeek-V3,同架构)现在只剩「多节点 + 规模」两个维度,代码已证明。
✅ **跨节点(多节点)已通**(2 节点 V2-Lite,8 trainer@node0 + 8 inference@node1,16 ranks 跨节点):
cross-group 跨节点 rendezvous(host=SLURM_JOB_FIRST_NODE_IP,rank0 当 TCPStore master)、
driver 跨节点寻址(node1 经头节点 IP 连 node0 trainer)、跨节点 NCCL 传输全成立。
kjobctl 多节点机制:JOB_COMPLETION_INDEX(节点 index)+ SLURM_JOB_FIRST_NODE_IP(头节点)+
固定端口(SLURM_JOB_ID 跨节点不一致不能派生)。run_e2e_multinode.sbatch。
✅ **671B 计划离线 dry-run**(纯几何,mac/CPU 零 GPU):DeepSeek-V3 909 权重、
trainer128→vllm(dp4 tp8)、model 1342GB、受端总收 1601GB(1.2×)。
**结构完全成立:0 unsupported placement、0 divisibility 问题**——get_m2m_map 处理全部
671B 权重,128 整除干净。**真 fallback 只 embed_tokens+lm_head**(是 is_sharded 缺口、
本该 TP 切,671B 浪费 ~115GB);其余 425 个 replicate(q_a/kv_a/gate/norm)是 vLLM
设计上的必要广播,不是浪费。修正前文"embedding floor/is_sharded 砍峰值":准确说是
embed+lm_head 这两个,量级 115GB,不是笼统的"所有 replicate"。
余项(通往 128→32 671B):trainer 多节点 torchrun(>8 rank)、inference 多节点 vLLM
(ray / external_launcher)、671B 加载(tracer-init 测机制 / sharded loading 出真权重)、
节点预算(128→32 ≈ 20 节点)。多 replica、量化档、容错实测。

**M3 bench**
chain vs fanout A/B(`split_fanout` 开关)、窗口扫参、与旧 etha 对比。
**fanout 不是死代码,是有真实优势域的策略**:全双工下 chain 中继的额外 send 被
RX/TX 独立隐藏,两者都 ≈ 收端 bound(R+D);fanout 唯一劣势是 source TX 串行 K·D,
仅当 `K·D > R+D` 即 **K > 1 + R/D** 时 chain 才赢。RL 是收端 bound(R 大、单块
replicated 权重 D 小),crossover K 偏大(~≥4),**小 K 时 fanout 反而更优**(等价
bound + 广播负担甩给闲的 trainer 侧 + 无 pipeline-fill)。bench 实测这个 crossover。

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
