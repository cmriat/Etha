# 分布式张量传输示例

本示例演示了使用不同设备网格配置在训练和推理工作节点之间进行**P2P优化的分布式张量传输**。

## 🎯 核心特性

- **多维张量**: 使用4×4张量进行调试，而非简单的标量
- **设备网格支持**: 数据并行 vs 模型并行策略
- **P2P优化**: 基于放置配置自动进行张量分片/复制
- **相同架构**: 4个训练 + 4个推理工作节点（与weight_transfer相同）

## 🏗️ 架构

```
训练工作节点 (Ranks 0-3)          推理工作节点 (Ranks 4-7)
┌─────────────────┐                  ┌─────────────────┐
│   GPU 0-3       │                  │   GPU 4-7       │
│   设备网格       │─────P2P─────▶    │   设备网格       │
│   策略: DP/MP   │   传输           │   策略: DP/MP   │
│                 │                  │                 │
└─────────────────┘                  └─────────────────┘
```

## 🔧 分布式策略

### 1. 混合DP+MP (默认)
```bash
TRAINING_STRATEGY=hybrid_dp_mp INFERENCE_STRATEGY=hybrid_dp_mp
```
- **网格形状**: (2, 2) - 2×2网格
- **放置配置**: (Replicate(), Shard(dim=0))
- **行为**: 模型参数在网格维度0上复制，在维度0上分片
- **用例**: 结合参数复制和行分片的混合并行

### 2. 纯模型并行
```bash
TRAINING_STRATEGY=pure_mp INFERENCE_STRATEGY=pure_mp
```
- **网格形状**: (4,) - 1×4线性
- **放置配置**: (Shard(dim=1),)
- **行为**: 模型参数(4,4)在隐藏维度上按列分片
- **用例**: 跨GPU的大型模型分区

## 🚀 快速开始

### 1. 启动所有进程
```bash
# 默认：双方都使用混合DP+MP
./launch.sh

# 通过环境变量自定义策略
TRAINING_STRATEGY=pure_mp INFERENCE_STRATEGY=pure_mp ./launch.sh

# 混合策略
TRAINING_STRATEGY=hybrid_dp_mp INFERENCE_STRATEGY=pure_mp ./launch.sh
```

### 2. 监控日志
```bash
# 训练进度
tail -f $PIXI_PROJECT_ROOT/prototyping/distributed_tensor_transfer/logs/train.log

# 推理进度
tail -f $PIXI_PROJECT_ROOT/prototyping/distributed_tensor_transfer/logs/inference.log

# Agent活动
tail -f $PIXI_PROJECT_ROOT/prototyping/distributed_tensor_transfer/logs/agent.log
```

### 3. 尝试不同配置
```bash
# 停止当前运行 (Ctrl+C)

# 双方都使用纯模型并行启动
TRAINING_STRATEGY=pure_mp INFERENCE_STRATEGY=pure_mp ./launch.sh

# 使用混合策略启动
TRAINING_STRATEGY=hybrid_dp_mp INFERENCE_STRATEGY=pure_mp ./launch.sh
```

## 📊 预期输出

### 训练工作节点日志
```
[Training Worker 1234] [INFO] 分布式训练工作节点启动...
[Training Worker 1234] [INFO]   本地排名：0
[Training Worker 1234] [INFO]   Agent排名：0
[Training Worker 1234] [INFO]   CUDA设备：cuda:0
[Training Worker 1234] [INFO]   分布式策略：hybrid_dp_mp
[Training Worker 1234] [INFO] 设备网格形状：(2, 2), 放置：['Replicate()', 'Shard(dim=0)']
[Training Worker 1234] [INFO] 排名0：创建分布式张量，形状torch.Size([4, 4])
[Training Worker 1234] [INFO] 排名0：本地张量形状torch.Size([2, 4])
[Training Worker 1234] [INFO] [train rank=0] step=0 full_tensor=
[Training Worker 1234] [INFO] tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.]])
```

### 推理工作节点日志
```
[Inference Worker 5678] [INFO] 分布式推理工作节点启动...
[Inference Worker 5678] [INFO]   本地排名：0
[Inference Worker 5678] [INFO]   Agent排名：4
[Inference Worker 5678] [INFO]   CUDA设备：cuda:0
[Inference Worker 5678] [INFO]   分布式策略：pure_mp
[Inference Worker 5678] [INFO] 设备网格形状：(4,), 放置：['Shard(dim=1)']
[Inference Worker 5678] [INFO] 排名0：创建分布式张量，形状torch.Size([4, 4])
[Inference Worker 5678] [INFO] 排名0：本地张量形状torch.Size([4, 1])
[Inference Worker 5678] [INFO] [inference rank=0] full_tensor=
[Inference Worker 5678] [INFO] tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.]])
```

## 🔍 与weight_transfer的关键区别

| 特性 | weight_transfer | distributed_tensor_transfer |
|---------|----------------|----------------------------|
| 张量形状 | (,) 标量 | (4, 4) 2D张量 |
| 设备网格 | 无 | (2,2) 混合DP+MP 或 (4,) 纯MP |
| 放置配置 | 无 | Replicate/Shard |
| 传输方式 | 简单发送/接收 | P2P优化 |
| 策略 | 单GPU | 混合DP+MP / 纯MP |

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

### P2P注册
```python
# 注册设备网格和放置配置以进行优化
handler = client.register_pair(
    pair_name=PAIR_NAME,
    local_name=LOCAL_NAME,
    remote_name=REMOTE_NAME,
    tensor=distributed_tensor,
    device_mesh=device_mesh,
    placements=tuple(placements),
)
```

## 🧪 测试不同配置

### 测试混合DP+MP
```bash
# 训练：混合复制 + 行分片
# 推理：混合复制 + 行分片
TRAINING_STRATEGY=hybrid_dp_mp INFERENCE_STRATEGY=hybrid_dp_mp ./launch.sh
```

### 测试纯模型并行
```bash
# 训练：按列分片参数
# 推理：按列分片参数
TRAINING_STRATEGY=pure_mp INFERENCE_STRATEGY=pure_mp ./launch.sh
```

### 测试混合策略
```bash
# 终端1：使用混合DP+MP进行训练
TRAINING_STRATEGY=hybrid_dp_mp python train.py

# 终端2：使用纯模型并行进行推理
INFERENCE_STRATEGY=pure_mp python inference.py
```

## 🔍 调试技巧

1. **检查P2P映射生成**：在agent日志中查找"Generated P2P map"
2. **验证网格一致性**：检查"mesh/placement validation passed"消息
3. **监控传输类型**：查找"Using optimized P2P transfer" vs "simple send/recv"

## 🚨 常见问题

- **NCCL错误**：distribute_tensor(src_data_rank=None)
- **端口冲突**：如果默认端口被使用，在launch.sh中更改BASE_PORT

## 📝 代码结构

```
prototyping/distributed_tensor_transfer/
├── agent.py              # Agent进程（8个rank）
├── train.py              # 训练工作节点（4个rank）
├── inference.py          # 推理工作节点（4个rank）
├── common.py             # 共享配置和工具函数
├── launch.sh             # 启动脚本
└── README.md             # 本文档
```

## 🔧 环境变量

- `TRAINING_STRATEGY`: 训练策略 (`hybrid_dp_mp` 或 `pure_mp`)
- `INFERENCE_STRATEGY`: 推理策略 (`hybrid_dp_mp` 或 `pure_mp`)
- `AGENT_RANK_OFFSET`: 推理工作节点的Agent排名偏移量 (默认: 4)