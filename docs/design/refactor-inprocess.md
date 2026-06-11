# Etha 重构:权重同步的第一性分解

**不同引擎之间同步权重(trainer → inference engine),差异只来自四类:name、parallel、
仿射变换、非仿射变换。** 整个 Etha 的设计,就是把这四类各归各位——**发送侧不碰引擎内部就能
产生的,归发送侧;只有引擎内部 kernel 才能产生的,归引擎。**

| 类 | 性质 | 例 | 谁做 |
|---|---|---|---|
| **name** | 字符串映射 | q/k/v ↔ `qkv_proj`、`w13`↔`gate_up` | 每个引擎提供 HF↔自己 的映射(vLLM 即 `WeightsMapper`;HF 名当枢纽)|
| **parallel** | placement→placement,值不变 | FSDP/EP/TP → 引擎 TP/EP | **Etha**:m2m reshard,never full |
| **仿射变换** | 轴级 view(offset/stride/permute)| fuse、pad、transpose/permute(含 per-model 格式归一化)| **Etha** placement m2m + consumer 用 `get_layout` 注册仿射 view |
| **非仿射变换** | 真 repack / 算新值 | swizzle/interleave、量化 | **引擎**:`process_weights_after_loading` |

> 一条分界线:**仿射的(parallel + fuse/pad/transpose/permute)发送侧不碰引擎内部就能产生 → Etha 做;
> 非仿射的(swizzle、量化)造它得复刻引擎 kernel → 引擎做。**
> **这条线不是我们硬切的——vLLM 内部本来就这么分**:`load_weights`(把 checkpoint 布局归一到
> 标准逻辑布局:narrow + fuse + 轴级 transpose/permute,全仿射)/ `process_weights_after_loading`
> (转 kernel-format:swizzle + 量化,非仿射)。Etha 接前段,引擎留后段。
>
> trainer **发它手里已有的 dtype**:bf16 训练就发 bf16(量化留引擎);**原生 fp8 训练(同 recipe)
> 就直接发 fp8+scale**(它本来就有,不引入额外 quant kernel)。原则:trainer 绝不为迁就引擎
> 去跑额外 quant kernel。


这个分解决定了一切:Etha 只做 parallel 的 placement m2m;仿射变换由 consumer 用 `get_layout`
注册成 fused param 的轴级 view,Etha reshard 进去即可(现成 `Chunk` 的 `tensor`(view)+
`prepare(contiguous)` 就能落)。非仿射全留引擎自己的加载后处理。引擎只需声明式地暴露
name / parallel / 仿射 view 三样元数据(下文),Etha 就能 never-full 地把权重送到位,
零硬编码、零 trace、不绑任何引擎内部。

## 范式:任意两个引擎间的 weight sync

前提:**每个引擎(训练 or 推理,Megatron / FSDP / vLLM / SGLang / 自研)各自实现同一套四类
抽象**——把"自己的权重表示"映射到四个 universal 中间表示(name→HF 名、parallel→DTensor
placement、仿射→HF 逻辑布局;非仿射无 universal,留各自):

```python
class EngineWeightAPI(Protocol):
    # ① name:HF 标准名 ↔ 本引擎名(融合分解 + rename;没列的默认 identity)
    name_map: NameMapper          # vLLM = WeightsMapper + packed_modules_mapping;Megatron = weight_converter
    # ② parallel:每个逻辑权重的分布 —— 只声明 (各维 size, placements)(多轴 EP/TP)
    #    真 DeviceMesh 由 driver 用 arange(base_rank,…).view(shape) 本地重建,不跨 world 传
    def get_sharding(self, hf_name) -> tuple[MeshShape, Placements]: ...
    # ③ 仿射 layout:逻辑权重落在本引擎 fused param 的哪一段(轴级 view)
    #    主体是 fuse —— q→qkv[0:nq]、gate→w13[0:h],就是个 offset(每个引擎都融合 qkv/gate_up/MoE w13);
    #    少数模型再带一个轴 permute(Conv1D .t() / QKV 去交错 / rotary),permute=None 即纯 fuse
    def get_layout(self, hf_name) -> list[ViewSpec]: ...
    # ④ 非仿射:收到逻辑权重后引擎内部 swizzle/quant → kernel-format(仅 receiver 用;
    #          训练侧没有——训练不 swizzle)
    def process_after_load(self, module) -> None: ...
```

有了这套,任意 `src → dst` 同步就是**纯粹在四类抽象上的操作**,Etha 不认识任何具体引擎:

```python
def weight_sync(src: EngineWeightAPI, dst: EngineWeightAPI, transport):
    # ===== INIT(一次性,plan 缓存复用)=====
    # ① name:两端各自归一到 HF 名,按 HF 名 join(只配两端都有的 → ⑤ 存在性自动排除)
    pairs = join_by_hf_name(src.name_map, dst.name_map)         # 每引擎一份映射,N+M 不是 N×M

    plan = []
    for hf_name in pairs:
        # ② parallel:两端 placement → m2m reshard(纯 placement,shard→shard,never full)
        chunks = m2m_plan(src.get_sharding(hf_name), dst.get_sharding(hf_name))
        # ③ 仿射:落点绑到 dst staging 上的轴级 view(fuse/transpose 由 ViewSpec 编码,Etha 不感知)
        bind_dst(chunks, dst.staging, dst.get_layout(hf_name))
        plan += chunks
    cache(plan)                                                 # placement/layout 不变 → plan 复用

    # ===== 每轮 sync =====
    run(plan, transport)                                        # src 读自己的 shard → dst staging 的仿射 view,never full
    dst.process_after_load(dst.modules)                         # ④ 非仿射:swizzle/quant → kernel(per-rank 本地)
```

整套设计的根就在这段:**四类各有一个 universal 中间表示,两端各转一跳**——
① name → **HF 标准名**(join 主键);② parallel → **DTensor placement**(m2m never-full 的输入);
③ 仿射 → **HF 逻辑布局**(`ViewSpec`,dst 落点);④ 非仿射 → **无 universal,留各自引擎**
(`process_after_load`)。

所以 **N 个 trainer × M 个 inference 不需要 N×M 套两两适配,只要每个引擎实现一次这四类(N+M)**;
Etha 作为中间只在 universal 表示上算 m2m + 落 view,**零引擎特定逻辑、零硬编码、never full**。
FSDP-native trainer "免费",是因为它的表示恰好已是这几个 universal 形式;换 Megatron 就得补
①②③ 的 converter(④ 训练侧天然没有)。下文 `## 理想的 in-process 用法` 是这套范式在 in-process
disaggregated 下的具体落地(由 driver 经 `collective_rpc` 编排)。

## 架构理念

围绕一个独立的 **comm 核心**(planner + `Chunk` + `chunk_comm`)重构 Etha:核心只负责
data plane,control plane 完全外包给 driver。Etha 退化成一个库,而非一个自带进程模型
和协调逻辑的系统。默认且唯一主路径是 **in-process disaggregated**——极简、已验证
(DeepSeek-671B 全模型同步在 32-GPU 推理部署下 ~1s)。

> Etha 的 data plane 永远是 peer-to-peer(跨卡走可插拔 transport——默认 NCCL,也可
> NIXL / Mooncake;同卡走 CUDA IPC;无中心中转)。
> 它的 control plane 是 single-controller(由 driver 统一发令;worker 之间
> 绝不互相轮询、选举或心跳)。

comm 核心不知道、也不关心控制流从哪来,也不绑定具体 transport——它只接受 driver
算好/下发的 plan,经 `Transport` 接口执行传输。

## 四类各归各位:仿射给 Etha,非仿射给引擎

开头那张表是全文的脊柱。这一节把它落实。

### name + parallel:Etha 的本行

- **name**:**HF 标准名是 universal 枢纽**。每个引擎(训练 or 推理,Megatron / FSDP / vLLM /
  SGLang)各自提供一份 **HF ↔ 本引擎名** 的映射;同步时两端各转一跳到 HF 名,就对上了。
  - trainer 侧:如 Megatron `decoder...linear_qkv.weight` ↔ HF `...q/k/v_proj.weight`
    (verl 的 `weight_converter.py` 就是这个);
  - 推理侧:name mismatch 有两种,vLLM 用**两个独立机制**各管一个——
    **`WeightsMapper`**(1→1 改名:前缀/子串/正则替换)+ **`packed_modules_mapping`**
    (1→多 融合分解:`qkv_proj` ← q/k/v)。两者互不调用,都是**差异表**(没列的默认 HF=vLLM,identity);
  - → **配对靠 HF 名当主键,不靠位置**。N 个 trainer × M 个 inference 不需要 N×M 套两两映射,
    只要**每个引擎一份 HF↔自己**,N+M 套搞定。Etha 核心只认 HF 名,不懂任何具体引擎命名。
  - ⚠️ trainer→HF 这跳是 **trainer 框架特有**的、需要手写 converter(命名是各框架自己的事),
    placement/layout 帮不上——这是 name 类的固有成本,谁都逃不掉(verl 也手写)。
- **parallel**:**universal 中间表示是 DTensor `(DeviceMesh, placement)`**。每个引擎(训练 +
  推理)都得提供自己的 placement——
  - trainer 是 **torch FSDP2 / DTensor-native 时刚好免费**(参数本就是带 placement 的 DTensor,
    直接读);**Megatron 就不免费**——它的 TP/PP/EP 分片是命令式写在自己代码里的(和 vLLM 的
    fuse 藏在 weight_loader 里同构),placement 表达不出来。要在它上面做 never-full reshard,
    得新写一个 Megatron-sharding→placement 的 converter,或 Megatron 上游暴露 placement。
    (verl 现在**没有**这个——它的 `weight_converter.py` 注释明写 "not including resharding",
    并行靠 `all_gather` 成 full_tensor 绕过,即 verl 那条被诟病的 full_tensor 路线。)
  - 推理侧用 `get_sharding` 暴露(多轴 EP/TP);
  - Etha 的 **m2m** 在两端 placement 间算 reshard,**Shard→Shard all-to-all,never full**。

> **统一规律**:四类里前三类(name / parallel / layout)都是"每个引擎声明 **自己 ↔ 一个 universal
> 中间表示** 的映射"——name↔**HF 标准名**、parallel↔**DTensor placement**、layout↔**HF 逻辑布局**。
> FSDP-native trainer "什么都不用做",只是因为它的表示**恰好就是这三个 universal 形式**;换
> Megatron,这三类都要 converter,和推理侧要 `WeightsMapper`/`get_sharding`/`get_layout` 完全对称。
> 非仿射类没有 universal(留引擎)。

### 仿射 vs 非仿射:② 这一刀是全文关键

`DeviceMesh + Placement` 只描述 parallel(分布),**描述不了变换**(tensor 自身重排)。
但"变换"内部要再切一刀——**这刀决定谁做**:

| 变换 | 是 view 吗 | 谁做 |
|---|---|---|
| **仿射**:fuse、pad、transpose/permute | ✅ 轴级 offset/stride/permute 能描述 | consumer 用 `get_layout` 注册仿射 view + **Etha** placement m2m |
| **非仿射**:swizzle/interleave、量化 | ❌ 没有 stride 能描述 / 算新值 | **引擎**:`process_weights_after_loading` |

- **仿射不是 Etha 的"一个操作",是 consumer 注册的 view**——Etha 始终只做 placement m2m
  (`get_m2m_map` 吃 mesh+placement),fuse/transpose 全靠 consumer 用 `get_layout` 把逻辑权重
  注册成 fused param 的轴级 view(q→`qkv[0:nq]`、Conv1D→`.t()`、QKV 去交错→轴 permute),
  Etha reshard 进去、**不感知它是 fused/转置的一段**。fuse/pad 是纯 offset(免费);
  transpose/permute 是 strided view → `prepare(contiguous)` 落一次 copy(非免费,但 Etha 现成);
- **非仿射(swizzle)Etha 干不了**:例 —— 物理顺序从 `0,1,2,3,4,5` 变成 `0,1,4,5,2,3`
  (元素 4 和 2 互换),**没有任何 stride 能表达**,只能真 repack;量化是算新值,同理;
- **发送侧产不出这些字节**:trainer 从不 swizzle,造它=复刻引擎 kernel(per-kernel、
  version-locked,反模式)。**所以非仿射无条件留引擎。**

> **trainer 发它手里已有的 dtype**:bf16 训练发 bf16(量化留引擎,量化模型 2× 带宽换简洁,
> 权重同步非热路径,值);**原生 fp8 训练(同 recipe)直接发 fp8+scale**(1× 带宽,trainer
> 天然就有这些字节)。不变的原则:**trainer 绝不为迁就引擎引入额外 quant kernel**。

### 决定性事实:load=仿射 / process=非仿射,是引擎自己画的边界

全量扫过 vLLM 全部 layer module + 279 个模型的 `load_weights`/`weight_loader`,坐实**三档**:

- **标准路径(Llama/Qwen/Mistral/GLM/Gemma + 多数 dense/MoE)**:`linear.py`(Column/QKV/Merged)
  与 `vocab_parallel_embedding.py` 的 loader = 纯 narrow(TP shard)+ fuse(offset)+ pad;量化只
  调 offset/packing(`adjust_marlin_shard`、`packed_factor`),数据操作仍是 narrow+copy,不 transpose
  不算值。所有量化/swizzle 数学在 `process_weights_after_loading`(scale 计算、`gptq_shuffle`、
  marlin/cutlass repack)——量化方法**无一自定义 weight_loader**,全复用 linear 的纯 narrow loader。
  这个 hook 自 FP8 引入(PR #4118),docstring 即 "process the weight after loading, e.g. transpose
  for computation",本就是**关注点分离**:load=加载,process=计算准备。
- **约 20 个模型 + MoE 量化路径**:`load_weights` 里多一步 **per-model 轴级格式归一化**,把该
  checkpoint 的 on-disk 布局重排成标准逻辑布局——全是 value-preserving 的轴转置/permute:
  Conv1D `.t()`(`gpt2`/`jais`)、fused-QKV 去交错(`gpt_neox`/`bloom`/`falcon`/`persimmon`)、
  rotary Q/K permute(`llama4`/`fairseq2`)、MoE expert `.transpose(1,2)`(`dbrx`/`aria`/`qwen3_vl_moe`)、
  WNA16/compressed-tensors MoE 的 `is_transposed→.t()`(`fused_moe/layer.py`,layer 级)。
  **这些仍是仿射、仍在 load(不在 process)**——只是 load 侧的仿射不止 fuse,还含 per-model 归一化;
  统一由 consumer 的 `get_layout` view spec(offset+permute)表达,对 Etha 透明。
- **极少数(scope out)**:`plamo2` 把 RMSNorm 单位偏移 `+= 1.0` 折进权重、`rnj1` 类似 `-=`——
  这是 load 阶段的**算值(非仿射)**,view 表达不了。数量极小、是 norm offset folding 的模型怪癖,
  **本设计明确不覆盖**(这类 param 走真 load 数学,不归 Etha bypass)。

→ **「仿射 / 非仿射」精确对上引擎「load_weights / process_weights_after_loading」**:load 侧全是
仿射(shard + fuse + 轴级归一化),process 侧全是非仿射(swizzle + 量化)。多挖出的那些 transpose
不是反例,恰恰**坐实**这条边界——它们是 load 侧多出来的仿射,落在边界该在的一侧。Etha 接前段
(仿射,view + m2m),引擎留后段(非仿射)。

### 落地:Etha bypass weight_loader,直落为默认,staging 为例外

```
直落(默认):布局无断裂的配置 —— bf16→bf16,或原生 fp8 直发 + plain kernel
trainer 逻辑权重 ──m2m──▶ 常驻 kernel param 的仿射 view(process 是 no-op)
   零 staging、零分组,全模型 chunks 一把进 chunk_comm,流水完整

staging(例外):process 真干活的配置 —— 收 bf16 由引擎量化,或 swizzle kernel
trainer 逻辑权重 ──m2m──▶ 临时 staging(HF 布局)──process──▶ kernel param
   裸层循环:for layer: staging=empty(); chunk_comm(layer_chunks); process(layer)
```

- **Etha recv 进 consumer 注册的仿射 view → 根本不调 weight_loader**(它的
  narrow + fuse + 轴级归一化正是 view + m2m 替掉的);
- **staging 是纯 HF 布局**(gate 永远前半,静态)——backend 特有的 swap([gate;up]↔[up;gate])
  发生在 `process_weights_after_loading` 内部从 HF→kernel 时,**Etha 看不见、不感知 backend**;
  staging 还隔离了 process 的 `replace_parameter`(kernel param 地址每轮变,staging 不受影响);
- **staging 档的执行就是裸层循环**:每层临时 `torch.empty`(caching allocator 复用),同流顺序
  提交,process 不与传输重叠——代价 ~1ms/层 × 层数 ≈ +10% 同步时间,先付着;K 槽 + cuda event
  的重叠流水是存档优化,bench 实测疼了再捡。staging 大小由 plan 导出(本 rank 该层 shard 字节,
  百 MB 级,**不是全局层大小**);IPC/NIXL 要求的固定地址 buffer 是 transport 的私事(它的 init
  持有并注册),不进核心概念;
- **`process_weights_after_loading` 是 per-rank 本地的**:扫过全部 96 个实现,无一 `all_gather` /
  `all_reduce` / `full_tensor`(collective 只在 `forward()` 和 MLA 前向;quant 里的 `tp_size` 只用于
  `create_weights` 的 block 对齐校验,不 gather)。所以 **Etha 送 local shard、process 就地跑 →
  端到端 never-full**;对比标准 `load_weights` 设计成收**完整 HF tensor** 再 narrow(verl 才被迫
  `all_gather` 成 full)。唯一要点:online per-tensor 量化的 scale 按本地 shard `.max()` 算(per-rank),
  是 vLLM 本来的行为,Etha 不引入新的训推不一致。

### 引擎只需暴露三样声明式元数据

```python
packed_modules_mapping                                       # name:qkv ← q/k/v(现成)
get_sharding("q_proj") -> (mesh_dim_sizes, placements)       # parallel:只声明 shape+placements(多轴 EP/TP)
   # q_proj -> ((tp,), [Shard(0)]);  w13 -> ((ep,tp), [Shard(0),Shard(1)])
get_layout("q_proj") -> [ViewSpec(fused="qkv_proj", offset=0, sub_shape=..., permute=None)]
   # 轴级仿射 view(offset + reshape + permute);fuse/pad 是 permute=None 的特例,
   # Conv1D/QKV去交错/rotary 是带 permute 的轴转置 —— 一个 spec 统一覆盖
```

- 三样都是**声明式元数据**(KB 级、init 可查、与显存无关),不是真 tensor;
- `get_sharding` 跨 world 给 trainer 算 m2m,但**只传 `(各维 size, placements)`,不传 DeviceMesh**
  (DeviceMesh 含 PG,跨 world 传不了):真 mesh 由 driver 本地 `arange(base_rank, …).view(shape)`
  重建(`get_m2m_map` 的 `distribute_tensor` + `.mesh` 要真 mesh);`base_rank` 按 world 布局赋
  (trainer 占 `0..T-1`、推理占 `T..`)。**前提:引擎 global rank 按 mesh 维序行优先排**(vLLM
  `(dp,tp)`、trainer `(dp_replicate,dp_shard[,ep])`,helper 即编码此约定)——非行优先的引擎才需真传 rank 张量;
- `get_layout` 推理侧本地把接收 buffer 注册成 HF staging 的轴级 view(execution 时
  `staging.narrow(...).view(sub_shape).permute(perm)` 现切);返回 list 兼容罕见多段;
- offset/permute 是 **HF 布局的静态量**(不含 backend swap,swap 在 process 内);
- 落点始终是已有的 HF staging,**不另开一份 buffer**:**fuse/pad 是连续 view → recv 直达 staging
  (零额外 copy)**;**transpose/permute 是非连续 view → wire 要连续,`prepare` 开一个 per-chunk
  连续临时块、recv 进去,`finalize` 里 `staging_view.copy_(tmp)` 把转置 scatter 进 staging**。
  复用 Etha 现成 prepare/finalize;转置那次 copy 折进 staging 写入(on-GPU memory-bound,便宜),
  只多一个 wire 临时块,不是多一份 staging。

→ 这三样替代了现状 `_convert_vllm_state_dict` 那张硬编码表(单模型 / 手抄融合 / 版本脆),
也比对 weight_loader 做 trace 简单:**仿射 offset 声明即可,非仿射本就在 process 里、引擎干**。

> ⚠️ `get_m2m_map` 内部的 `full_tensor` 只作用在 LCM 大小的 middle 指纹张量(算 plan 的
> 一次性小开销),**不是每轮 full 真权重**——真权重走 chunk,never full。两个"full"别混。

## colocate vs disaggregated:数据走哪

真正的分界轴是一个 chunk 的 source 和 destination 是否在 **同一张物理卡** 上:

| 放置 | transport |
|---|---|
| 同卡(跨进程) | CUDA IPC(device-to-device copy) |
| 不同卡 / 跨节点 | 跨卡 transport(P2P send/recv,默认 NCCL,可插拔——见下节) |

同卡始终走 IPC,不走跨卡 transport。尤其用 NCCL 当跨卡 transport 时,同卡两进程之间
用 NCCL 是不安全的:无 MPS 时有 SM 争抢;且一个 NCCL communicator **不能把同一个
CUDA device 复用为两个 rank——会 hang**。所以同卡 chunk 必须走 IPC。planner 按物理
放置给每个 chunk 打标,`chunk_comm` 逐 chunk 分派(IPC / 跨卡 transport)。

## Driver 装配:从部署拓扑到 (mesh, placements)

plan 的输入是两端的 sharding 声明;声明从哪来,是 driver 的三件装配活:

1. **rank 记账**:cross-world 编号——trainer 占 `0..T-1`,各推理 replica 依次占后续
   区段(`create_cross_group` 的输入,与 mesh tensor 同一编号体系);
2. **mesh 拼装**:一个统一 mesh,维序 = 物理 rank 序(行优先,tp 最内):
   `arange(R*dp*tp).view(R, dp, tp)`,R 是框架层 replica 维;
3. **placement 拼装**:per 权重类查下表,所有层共用同一个 mesh。

两端的纪律相反:

- **trainer(DTensor-native)**:placement 必须从 **`param.placements` 读**,不许手填。
  FSDP2 叠在 EP/TP 之上的真实布局是 `_StridedShard`(物理上内层先切、FSDP 后切),
  手填标准 `Shard` 表与实际布局不符 → 静默错数据。Etha 原生支持规范形态的
  `_StridedShard`(FSDP2 总是规范填法)。
- **推理(vLLM,非 DTensor)**:layer 对象只有 TP/EP 的命令式知识(`tp_rank`/`output_dim`
  藏在 weight_loader 里);**Replicate 维(replica×dp)只存在于部署拓扑中**,引擎对象上
  查不到——必须由 driver 拼。

三层 "DP" 角色各不同,这是手填 placement 最易错处:

| 层 | 独立性 | 对 MoE 权重 |
|---|---|---|
| 多 replica(框架层,如 verl `num_replicas`) | 真独立(各自调度) | `Replicate` |
| vLLM 原生 `data_parallel_size` | **lockstep**(EP 的 all-to-all 横跨 DP,空 batch 也要 dummy 陪跑) | EP on 时是**切分维** |
| TP | 组内协同算同一 batch | 切分维 |

vLLM 的 placement 映射(mesh `(R, dp, tp)`):

| 权重 | placements |
|---|---|
| Column 类(qkv/gate_up/embed) | `(Replicate, Replicate, Shard(0))` |
| Row 类(o_proj/down) | `(Replicate, Replicate, Shard(1))` |
| MoE w13/w2,**EP on** | `(Replicate, Shard(0), Shard(0))` —— dp、tp 嵌套切 expert 维;`ep_rank` 即 (dp,tp) 的行优先展平,嵌套切分 ≡ 一维 EP |
| MoE w13 / w2,EP off | `(Replicate, Replicate, Shard(1))` / `(R, R, Shard(2))` |
| norm / router | 全 `Replicate` |

- **EP 是"借格子",不是新 mesh 维**:`enable_expert_parallel` 时 expert 整只归属(vLLM
  源码:"In EP, each device owns a set of experts fully. There is no tensor
  parallel"),EP 吞掉 dp×tp 全部格子,MoE 内 TP 强制为 1——vLLM 主线没有 EP×ETP
  (Megatron trainer 侧存在,`(Replicate, Shard(0), Shard(1))` 即可表达)。
- **PP 不进 placement**:它切的是权重集合(stage 内每层完整)——per-param 的 target
  mesh 只填持有该权重的 rank,与"只同步两端都有的"是同一机制。
- **CP**:PCP 对权重 = Replicate;DCP 复用 TP 组的 GPU,权重照 TP 切,无新维。
- ⚠️ **placement 描述的是引擎的物理现状,而现状随版本(含 bug)漂移**:vLLM #36222 曾把
  非 EP MoE 的 dp 错误折进 flatten TP(权重物理上被错切)——装着该版本就得按错切的
  现状声明才能传对。根治是引擎自己暴露 `get_sharding`(声明与实现同源);在那之前,
  这张表要随 vLLM 版本核对。

进程与依赖的分界一句话:**元数据可以跨进程搬,计算跟着数据走**。`get_sharding`/
`get_layout`/process 触发跑在 vLLM 自己的进程里(经 collective_rpc),产出 KB 级纯数据
经 driver 给 trainer——trainer 进程永远只依赖 torch;而量化这类对 GB 级权重的计算
没法这么拆,想用引擎的代码就得把依赖带进 trainer(故发端量化仅 colocate 免费)。

### trainer PP:per-param mesh 的常态,零特判

trainer 开 PP 时不同权重的 source mesh 不同(stage s 只持有自己层区间的权重)——
这就是 per-param plan 的本义,planner/execution 零改动。装配也自动:PP 不出现在
DTensor 里,`param.device_mesh` 就是该 stage 的 dp/tp mesh(全局 rank 子集),逐参数
读即得。两个配套点:

- **tied embedding**:首尾 stage 各持一份同名权重 → driver 对同名多源去重任选其一;
- **`route_idx` 全局重编**:`m2m_to_chunks` 的 route_idx 是 per-param 的,多权重 chunks
  拼给一次 `chunk_comm` 前由 driver 按全局清单序重编为连续递增(几行 helper)——
  全局清单序全 rank 一致(按 name 排)即保证 FIFO 配对;
- **多 source 的并发与排序**:跨 stage 的 peer-pair 不相交,FIFO 安全性自动成立。
  按 name 排序意味着各 stage 依次发——**推理卡数 ≤ stage 卡数(常态)时收端入口
  受限,串行即带宽下界,无损**;仅当 trainer 出口 < inference 入口(小训练 × 大
  replica 群)时,把清单 sort key 换成 `hash(name)` 让各 stage 的边混进每个窗口,
  恢复 trainer 出口聚合——plan 时由两侧卡数判断,一行排序的事。

## control plane:single-controller,绝不常驻自协调

driver 是唯一同时横跨 trainer world 和 inference world 的实体。所有跨 world 的
metadata——IPC handle、plan、ncclUniqueId——都经过 driver。worker 之间绝不通过一个
常驻 store 互相协调。

要删掉的反模式(旧 `tensor_bus`):一个常驻 KVStore + 轮询 / leader 选举 / heartbeat。
这几样存在的唯一原因,是平等节点没有中心、只能互相猜:

| 机制 | 存在只因为 | single-controller 下 |
|---|---|---|
| 轮询 KVStore 等对面 ready | 没人统一发令,只能反复问 | driver 直接推一个 RPC;不用问 |
| leader 选举 | 平等节点要推举谁来建状态 | driver 就是 leader;无需选举 |
| heartbeat | 平等节点要互判死活 | driver 持有 worker 句柄;RPC 失败即信号 |

control plane 完全外包——Etha 只是一个库,由 driver / `collective_rpc` 下发
`etha_init` / `etha_transfer`。

## 与 vLLM 解耦:`CommBootstrap` 接口

(本节针对 **NCCL transport** 的 bootstrap——交换 ncclUniqueId 建 communicator。
NIXL / Mooncake 等 connection-based transport 有自己的连接建立,通常更简单,无需
uniqueId 集体交换,不受这里的约束。)

用 NCCL 当跨卡 transport 时,Etha 不能依赖 vLLM 的 `StatelessProcessGroup`(那会把
Etha 绑死在 vLLM 上;trainer 侧 / SGLang / 自研 engine 就用不了)。
`StatelessProcessGroup` 本身只是 "TCPStore 交换 ncclUniqueId + `ncclCommInitRank`"
的薄封装——是 torch.distributed / NCCL 的原生能力。

抽象一个最小接口;Etha 自带一个只依赖 torch.distributed 的默认实现:

```python
class CommBootstrap(Protocol):
    def create_nccl_comm(self, world_size, rank, host, port) -> NcclComm: ...

# 默认,零 vLLM 依赖:
class TorchStoreBootstrap(CommBootstrap):   # torch TCPStore + ncclCommInitRank
    ...
# 接 vLLM 时由 vLLM 注入它自己的(复用):
class VllmStatelessBootstrap(CommBootstrap):  # 包一层 StatelessProcessGroup
    ...
```

Etha 核心只认 `CommBootstrap`。默认 `TorchStoreBootstrap` → 依赖只有 torch + NCCL,
不绑任何 engine。接 vLLM 时由它注入自己的实现。

## 可插拔 transport:NCCL / NIXL / Mooncake / …

跨卡搬字节这一层(`chunk_comm` 里的 send/recv)应该接口化,和 `CommBootstrap` 并列成
第二个可插拔轴。它们替代的只是 **怎么搬**,不碰 planner(**搬什么 / shard→shard 映射**,
Etha 的核心 IP)。

```python
class Transport(Protocol):
    def send(self, buf, dst): ...
    def recv(self, buf, src): ...
    # NcclTransport(现状) / NixlTransport / MooncakeTransport / ...
```

候选(都不是更快,而是各自解 NCCL 的某些约束):

假定集群是标准 IB/RoCE + NVLink(非 AWS EFA),所以网络 fabric 不是区分维度——
几个库都支持 IB/RoCE + GPUDirect RDMA。真正的区分点是:有无 communicator、占不占 SM、
生态成熟度。

| 库 | 连接模型 | 占 SM? | memory 类型 | 成熟度 | 对 Etha |
|---|---|---|---|---|---|
| **NCCL**(默认) | communicator(成员固定,device 绑 rank) | P2P kernel spin SM | 仅 GPU | 最成熟,已验证 671B 1s | disaggregated 一卡一进程最稳;**colocate 同卡两进程会 device 复用 hang / SM 争抢** |
| **NIXL**(NVIDIA, UCX) | connection-based,无 communicator | host-initiated,NIC offload,少占 SM | GPU/CPU/NVMe/S3 | Dynamo/vLLM/SGLang/TRT-LLM 生态广 | **主力候选**:无 communicator → 跨 world / 同卡都顺;多 backend |
| **Mooncake Transfer Engine** | connection-based,无 communicator | NIC/copy-engine,少占 SM | GPU/CPU/NVMe | KV pool 生产验证 | 跨节点高吞吐强(topology-aware 多网卡聚合) |
| **NVSHMEM** | one-sided(GPU-initiated IBGDA) | **GPU 发起,占 SM/warp** | **仅 symmetric heap**(必须 `nvshmem_malloc`) | NVIDIA 成熟但偏 HPC | **基本出局**:只能在对称堆内通信,碰不了 torch 普通 tensor;要么拷进堆(多一次 copy + 固定 buffer)、要么接管权重分配(vLLM 不可能);one-sided 虽贴但约束太硬 |
| **Perplexity TransferEngine** ([arXiv:2510.27656]) | RDMA P2P | NIC offload | 仅 GPU | 新/研究;Etha blog 已引用 | 新,未广泛验证 |
| **UCCL** | RDMA P2P | NIC offload | 仅 GPU | 研究 | 候选之一 |

关键洞察:**NCCL 的 communicator 模型是 colocate 同卡那两个问题(device 复用 hang、
SM 争抢)的根源**。换成无 communicator 的 transport(NIXL / Mooncake / Perplexity TE):
connection-based → 没有 device-当两-rank 的约束;NIC-offload → 跨卡传输不 spin SM。
于是 **colocate 的跨卡 chunk 在 in-process 下也能干净跑**,同卡仍走 IPC。

内存模型(决定能不能直接接 Etha 已有的权重 tensor):
- **NCCL**:裸指针,`ncclSend(tensor.data_ptr())` 直接传,零预备——最省事;
- **NIXL / Mooncake**:**register 已有 region 一次**(权重还是 torch 分配的,不动;register 是 kernel call 有开销,故 init 时整块注册、之后传任意 sub-range)——对 Etha 友好;
- **NVSHMEM**:**必须 `nvshmem_malloc` 从对称堆分配**,碰不了 torch 普通 tensor——故出局。

即:除 NVSHMEM 外都是「对你已有的显存操作」,NVSHMEM 是唯一要迁就专属分配器的。

选型建议:
- **默认 NCCL** —— disaggregated、已验证、最稳;
- **想统一 colocate / 跨 world / 去掉 communicator 约束 / 异构 memory** → **NIXL**(生态最广、UCX 多 backend);
- **跨节点极致吞吐** → Mooncake TE(多网卡聚合)。

## 同卡 chunk 走 IPC:对 `Chunk` 的最小改动

Etha 已经有 `Transport.LOCAL`(同 rank self-copy,在 `prepare`/`finalize` 里做
NCCL-group 外的 `Tensor.copy_`)。跨进程的同卡 IPC,本质就是
"LOCAL self-copy,只是 source 是另一个进程 IPC 映射过来的 tensor"。所以:

- 给 `Chunk` 加一个 `remote_tensor: Tensor | None`(只在 **target** 侧用——IPC copy
  由 dst 单边驱动;source 进程每轮什么都不做);
- 在 `prepare` 里,`is_target` 分支有 `remote_tensor` 时从它(IPC 映射的 source)读,
  否则走原来的本地 recv 路径;
- `remote_tensor` 只填一次(chunk 注册一次、跨多轮 transfer 复用),init 时经 control
  plane 用 `ForkingPickler` 传 handle;
- 硬约束:source 权重必须 **in-place 更新**(不重新分配),否则 IPC 映射指向已释放的
  显存。

不需要 registry——Etha 的 chunk 是注册一次复用的,映射来的 tensor 跟着 chunk 活。

## 砍掉的复杂度(simplification)

重构里删两块对权重传输无用的东西:

**1. bucket(`Bucket` / `get_buckets` / `bucket_comm`)** —— bucket 是把很多小 chunk
攒成一个大 buffer 发,摊薄 per-op 开销。但大模型权重的 chunk 本就大(MoE up_gate
per-rank 几 GB),per-op 开销可忽略,bucket 收益≈0。碎参数(layernorm/bias)那点 op
数微不足道(占传输 <<1%),真疼了把它们拼进相邻大 chunk 即可,不值一套通用 bucketization。
→ `chunk_comm` 只走 chunk,不要 bucket。

**2. Partial 处理** —— `Chunk.source_partial_groups`、`prepare` 里的 all-reduce-before-cast、
bootstrap 的 `_create_partial_groups`、init 里"target 含 Partial 就 skip 方向"那套。
Partial placement 的语义是"未 reduce 的部分和"(出现在梯度/激活),**权重永远是
Shard / Replicate,从不是 Partial**。所以对纯权重传输这是 dead code。
→ 全删:`prepare` 退化成"读 slice + dtype cast",`Chunk` 字段少一半,init 少一大段。
(前提:Etha 只传权重。若还要传 optimizer state 等可能为 Partial 的东西,再保留。)

保留的(不是过度设计):
- **chunks.py 的两层**(抽象 M2M map + 每 tensor specialization)—— plan-once-apply-many,
  trace planner 算一次跨几百个权重复用,合并反而 N 倍贵;
- **prepare/finalize 的 dtype cast + 防 aliasing 的 clone** —— 正确性,非冗余;
- **多维 cell** —— 真实 placement 是多维 mesh(DP×TP×EP),必须。

## 模块结构

目录树体现「核心 planner + 可插拔 transport + 薄 API」,删掉 sidecar / kvstore:

```
src/etha/
  __init__.py            # 公开 API
  api.py                 # 薄库入口:etha_init / etha_transfer + export/import_ipc_handles
                         #   control plane 不在这里——由 driver 用 collective_rpc 串
  ir.py                  # Chunk(+remote_tensor)/Route/M2MMap/Endpoint(删 Bucket + Partial 字段)

  planner/               # 核心 IP(transport 无关)
    m2m_map.py           #   M2M:source placement + dst placement → chunk plan(纯 placement)
                         #   (原 comm/get_m2m_map.py;仿射 view 由 consumer 经 get_layout 注册,m2m 不感知)
    chunks.py            #   map_to_chunk_ops:把 m2m_map 套到 consumer 的仿射 view(原 comm/get_chunks.py)

  execution.py           # chunk_comm:逐 chunk 分派 IPC / 跨卡 transport(原 comm/comm_methods.py,无 bucket)

  transport/             # 可插拔轴:怎么搬字节
    __init__.py          #   Transport protocol
    nccl.py              #   NcclTransport(原 comm/transfer.py)
    ipc.py               #   同卡 CUDA IPC(remote_tensor,dst 单边)
    nixl.py / mooncake.py#   可选

  utils.py               # 杂项 + pg helpers + CommBootstrap 接口 + TorchStoreBootstrap(默认)
```

删除:`tensor_bus/` 整个(sidecar + 命令队列 + agent/client)、`kvstore/` 整个(常驻自协调)、
`comm/get_buckets.py`(bucket)。`bootstrap/` 不单列——就一个接口一个默认实现,塞 `utils.py`;
vLLM 的 bootstrap 实现接入时由 vLLM 侧注入,不在 Etha 仓。

| 当前 | 去向 |
|---|---|
| `comm/ir.py` | `ir.py`(删 Bucket / Partial 字段)|
| `comm/get_m2m_map.py` | `planner/m2m_map.py`(dst 用 vLLM `get_sharding`+`get_layout`)|
| `comm/get_chunks.py` | `planner/chunks.py` |
| `comm/comm_methods.py` | `execution.py`(无 bucket)|
| `comm/transfer.py` | `transport/nccl.py` |
| `comm/get_buckets.py` | **删** |
| `comm/utils.py` + `pg_utils.py` | `utils.py` |
| `tensor_bus/*` / `kvstore/*` | **删** |

## 理想的 in-process 用法(伪码)

```python
# ===== INIT(一次性,driver 编排,经 collective_rpc 扇出)=====

# consumer(vLLM worker):用 vLLM 声明式查询暴露 ①sharding + ②仿射 view
def etha_export_dst_spec(vllm_model):
    spec = {}
    for name in logical_weight_names(vllm_model):       # 经 packed_modules_mapping 分解
        spec[name] = (get_sharding(name), get_layout(name))   # 多轴 placement + 轴级 view spec(offset+permute)
    return spec                                          # KB 级,driver 收走

# provider(trainer):自知 mesh+placement,和 dst sharding 都是干净 placement → m2m
def etha_build_plan(trainer_mesh, trainer_placements, dst_spec, transport):
    chunks = m2m_plan(trainer_mesh, trainer_placements, dst_spec)  # 两端干净 placement;仿射 view 由 consumer 编码,m2m 落进去
    return chunks                        # 每 chunk 标 同卡→IPC / 跨卡→transport;缓存复用

def driver_init():
    dst_spec = collective_rpc(vllm_workers, etha_export_dst_spec)
    handles  = collective_rpc(vllm_workers, etha_export_ipc_handles)   # 同卡 chunk
    rpc(trainer, etha_build_plan, dst_spec, transport=NixlTransport)
    rpc(vllm_workers, etha_import_ipc_handles, handles)

# ===== 每轮 weight sync(driver 触发)=====
def driver_sync(version):
    parallel(
        rpc(trainer,      etha_send, version),
        rpc(vllm_workers, vllm_pause),                 # 暂停生成;etha 把新值直接原地写进权重 tensor(in-place)
        rpc(vllm_workers, etha_recv_and_load, version),
    )
    rpc(vllm_workers, vllm_resume)                      # KV 失效 → 重新 prefill 续推

# consumer:recv 进 HF staging 的轴级 view,再让 vLLM 做 swizzle/量化
def etha_recv_and_load(chunks, layer):
    for c in chunks:                                    # recv 直接落 consumer 注册的仿射 view(never full)
        if c.kind == "ipc": c.dst_view.copy_(c.remote_tensor[c.src_slice])
        else: transport.recv(c.dst_view, c.src_rank)    # dst_view = staging.narrow(...).view(sub_shape).permute(perm)
    process_weights_after_loading(layer)                # vLLM:swizzle + 量化 scale → kernel buffer
```

Etha 全程只是一组被 driver 调的函数,只做 placement reshard 进 consumer 注册的仿射 view——
**bypass weight_loader、无 sidecar、无 KVStore、无 full_tensor**;swizzle/量化全留 vLLM 的
`process_weights_after_loading`。

## 总结

| 问题 | 结论 |
|---|---|
| **第一性分解** | 权重同步差异只来自四类:**name / parallel / 仿射变换 / 非仿射变换**;存在性排除(只同步两端都有的)|
| **归属(一条线)** | 仿射(parallel + fuse/pad/transpose/permute)发送侧不碰引擎内部就能产生 → **Etha**;非仿射(swizzle + 量化)造它得复刻引擎 kernel → **引擎**(`process_weights_after_loading`)|
| **引擎内部即此分法** | 全量扫过全部 layer + 279 模型:`load_weights` 侧全仿射(narrow + fuse + per-model 轴级归一化:Conv1D `.t()`、QKV 去交错、MoE expert transpose),`process` 侧全非仿射(swizzle + 量化);`plamo2`/`rnj1` 的 norm `+=` 算值是极少数例外,**scope out** |
| 默认路径 | in-process、disaggregated、control plane 外包——极简、已验证 |
| **引擎需暴露** | 三样声明式元数据:`packed_modules_mapping`(name,现成)+ `get_sharding`(多轴 placement)+ `get_layout`(轴级仿射 view spec:offset+permute)|
| **Etha 直填** | bypass `weight_loader`(其 narrow+fuse+轴级归一化正是 view+m2m 替掉的),recv 进 HF staging 的轴级 view(transpose 落一次 strided copy),再调 `process_weights_after_loading`;staging 纯 HF 布局,backend swap 封在 process 内,Etha 不感知 |
| **量化策略** | trainer 发它已有的 dtype:bf16 训练发 bf16(量化留引擎,2× 带宽换简洁);原生 fp8 训练发 fp8+scale(1× 带宽,天然有)。原则:trainer 不为迁就引擎引入 quant kernel |
| control plane | 经 driver 的 single-controller;删掉常驻 KVStore / 轮询 / leader / heartbeat |
| bootstrap store | 一次性 rendezvous,不在 data path 上;影响轻微;或由 driver 中转 uniqueId |
| vLLM 耦合 | `CommBootstrap` 接口;默认 `TorchStoreBootstrap`,零 engine 依赖 |
| 同卡 transport | IPC(绝不用 NCCL):SM 争抢 + NCCL device 复用 hang |
| 跨卡 transport | 可插拔(`Transport` 接口):默认 NCCL;NIXL/Mooncake/Perplexity TE 无 communicator,可在 in-process 下跑 colocate |
| 为 IPC 改 Chunk | target 侧加 `remote_tensor`;复用 `Transport.LOCAL` 路径 |
```
