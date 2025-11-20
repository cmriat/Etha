# 分布式张量传输系统

使用 **P2P优化的分布式张量传输** 在训练和推理工作节点之间进行高效数据传输，支持多种设备网格配置。

## 🎯 核心特性

- **多维张量传输**: 支持复杂张量结构的P2P传输
- **设备网格优化**: 自动适配数据并行和模型并行策略
- **智能分片**: 基于放置配置自动进行张量分片/复制
- **高性能架构**: 4训练节点 + 4推理节点的分布式架构

## 🏗️ 系统架构

```
训练节点 (Ranks 0-3)              推理节点 (Ranks 4-7)
┌─────────────────┐                  ┌─────────────────┐
│   GPU 0-3       │                  │   GPU 4-7       │
│   设备网格       │─────P2P─────▶    │   设备网格       │
│   分布式策略     │   传输           │   分布式策略     │
│                 │                  │                 │
└─────────────────┘                  └─────────────────┘
```

系统包含三个组件：
- **Agent进程** (8个rank): 协调P2P传输
- **训练工作节点** (4个rank): 生成并发送张量
- **推理工作节点** (4个rank): 接收并使用张量

## 🚀 快速开始

### 基础用法

```bash
# 使用默认策略（混合DP+MP）
pixi run transfer_tensor

# 指定训练策略
TRAINING_STRATEGY=pure_mp pixi run transfer_tensor

# 同时指定训练和推理策略
TRAINING_STRATEGY=pure_mp INFERENCE_STRATEGY=hybrid_dp_mp pixi run transfer_tensor
```

### 分布式策略说明

#### 1. 纯模型并行 (pure_mp)
- **网格形状**: (4,) - 1×4线性
- **放置配置**: (Shard(dim=0),)
- **行为**: 模型参数按列分片
- **用例**: 跨GPU的大型模型分区

#### 2. 混合DP+MP (hybrid_dp_mp)
- **网格形状**: (2, 2) - 2×2网格
- **放置配置**: (Replicate(), Shard(dim=0))
- **行为**: 参数在网格维度0上复制，在维度0上分片
- **用例**: 结合参数复制和行分片的混合并行

## 📊 运行效果

### 训练节点输出
```
[Training Worker] [INFO] 分布式训练节点启动...
[Training Worker] [INFO]   全局排名：0
[Training Worker] [INFO]   Agent排名：0
[Training Worker] [INFO]   CUDA设备：cuda:0
[Training Worker] [INFO]   分布式策略：pure_mp
[Training Worker] [INFO] 设备网格形状：(4,), 放置：['Shard(dim=0)']
[Training Worker] [INFO] 创建分布式张量，形状torch.Size([4, 4])
[Training Worker] [INFO] 本地张量形状torch.Size([1, 4])
[Training Worker] [INFO] [train rank=0] step=0 full_tensor=
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.]])
```

### 推理节点输出
```
[Inference Worker] [INFO] 分布式推理节点启动...
[Inference Worker] [INFO]   全局排名：0
[Inference Worker] [INFO]   Agent排名：4
[Inference Worker] [INFO]   CUDA设备：cuda:0
[Inference Worker] [INFO]   分布式策略：pure_mp
[Inference Worker] [INFO] 设备网格形状：(4,), 放置：['Shard(dim=0)']
[Inference Worker] [INFO] 创建分布式张量，形状torch.Size([4, 4])
[Inference Worker] [INFO] 本地张量形状torch.Size([1, 4])
[Inference Worker] [INFO] [inference rank=0] full_tensor=
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.]])
```

## 🔧 高级配置

### 策略组合示例
```bash
# 训练使用纯模型并行，推理使用混合策略
TRAINING_STRATEGY=pure_mp INFERENCE_STRATEGY=hybrid_dp_mp pixi run transfer_tensor

# 双方都使用混合策略
TRAINING_STRATEGY=hybrid_dp_mp INFERENCE_STRATEGY=hybrid_dp_mp pixi run transfer_tensor

# 训练使用混合策略，推理使用纯模型并行
TRAINING_STRATEGY=hybrid_dp_mp INFERENCE_STRATEGY=pure_mp pixi run transfer_tensor
```

## 🛠️ 实现细节

### 设备网格创建
```python
# 为4个GPU创建2×2配置的网格张量
mesh_tensor = torch.arange(4).view(2, 2)
device_mesh = DeviceMesh("cuda", mesh_tensor)
```

### 分布式张量创建
```python
# 使用特定放置配置创建分布式张量
distributed_tensor = distribute_tensor(
    base_tensor,
    device_mesh,
    [Replicate(), Shard(dim=0)]
)
```

### Pair注册
一对(Device Mesh + Placement)构成一个Pair, 用于定义张量传输的源和目标配置。
```python
# 1. 初始化 pair（初始化行为）
client.init_pair(
    pair_name=PAIR_NAME,
    local_name=LOCAL_NAME,
    remote_name=REMOTE_NAME,
    device_mesh=device_mesh,
    placements=tuple(placements),
)

# 2. 注册 tensors（获取handler用于细粒度调度）
handler = client.register_tensors([
    (distributed_tensor.to_local(), PAIR_NAME)
])
```

## 🔍 调试技巧

1. **检查P2P映射生成**：在日志中查找"Generated P2P map"
2. **验证网格一致性**：检查"mesh/placement validation passed"消息
3. **监控传输类型**：查找"Using optimized P2P transfer" vs "simple send/recv"

## 🚨 常见问题

- **端口冲突**：确保端口39500-39502未被占用
- **权限问题**：确认对日志目录有写入权限

## 📝 项目结构

```
prototyping/distributed_tensor_transfer/
├── train.py              # 训练工作节点（4个rank）
├── inference.py          # 推理工作节点（4个rank）
├── launch.sh             # 启动脚本（已集成到pixi命令）
└── README.md             # 本文档
```