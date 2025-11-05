# 分布式模型传输系统

使用 **P2P优化的分布式模型传输** 在训练和推理工作节点之间传输完整的Qwen3-30B-A3B模型参数。

## 🎯 核心特性

- **完整模型传输**: 传输Qwen3-30B-A3B模型的所有参数
- **分布式策略**: 支持纯模型并行和混合DP+MP策略
- **参数验证**: 传输后自动验证模型一致性
- **高效架构**: 4训练节点 + 4推理节点的分布式架构

## 🏗️ 系统架构

```
训练节点 (Ranks 0-3)              推理节点 (Ranks 4-7)
┌─────────────────┐                  ┌─────────────────┐
│   GPU 0-3       │                  │   GPU 4-7       │
│   Qwen3-30B-A3B  │─────P2P─────▶    │   Qwen3-30B-A3B  │
│   分布式策略     │   传输           │   分布式策略     │
│                 │                  │                 │
└─────────────────┘                  └─────────────────┘
```

## 🚀 快速开始

### 基础用法

```bash
# 使用默认策略（训练混合DP+MP，推理纯MP）
pixi run transfer_model

# 指定训练策略
TRAINING_STRATEGY=pure_mp pixi run transfer_model

# 同时指定训练和推理策略
TRAINING_STRATEGY=pure_mp INFERENCE_STRATEGY=hybrid_dp_mp pixi run transfer_model
```

### 分布式策略

#### 纯模型并行 (pure_mp)
- **网格形状**: (4,) - 1×4线性
- **放置配置**: (Shard(dim=0),)
- **行为**: 模型参数按列分片

#### 混合DP+MP (hybrid_dp_mp)
- **网格形状**: (2, 2) - 2×2网格
- **放置配置**: (Replicate(), Shard(dim=0))
- **行为**: 参数在网格维度0上复制，在维度0上分片

## 📊 运行效果

### 训练节点输出
```
[Worker] [INFO] Distributed Training Worker starting...
[Worker] [INFO]   Global rank: 0
[Worker] [INFO]   Agent rank: 0
[Worker] [INFO]   CUDA device: cuda:0
[Worker] [INFO]   Distributed strategy: pure_mp
[Worker] [INFO] Device mesh: tensor([0, 1, 2, 3]), placements: (Shard(dim=0),)
[Worker] [INFO] Model created from pretrained model
[Worker] [INFO] ✅ Sent distributed model
```

### 推理节点输出
```
[Worker] [INFO] Distributed Inference Worker starting...
[Worker] [INFO]   Global rank: 0
[Worker] [INFO]   Agent rank: 4
[Worker] [INFO]   CUDA device: cuda:0
[Worker] [INFO]   Distributed strategy: pure_mp
[Worker] [INFO] ✅ Received distributed model
[Worker] [INFO] ✅ Distributed model matches golden model
```

## 🔧 高级配置

### 策略组合
```bash
# 双方都使用纯模型并行
TRAINING_STRATEGY=pure_mp INFERENCE_STRATEGY=pure_mp pixi run transfer_model

# 训练混合策略，推理纯模型并行
TRAINING_STRATEGY=hybrid_dp_mp INFERENCE_STRATEGY=pure_mp pixi run transfer_model
```

## 🛠️ 实现细节

### 模型分布式化
```python
# 为每个参数创建分布式张量
for name, param in model.named_parameters():
    dist_param = distribute_tensor(param, device_mesh, placements)
    model._parameters[name] = dist_param
```

### 参数验证
```python
# 传输后验证模型一致性
assert torch.allclose(param.full_tensor(), golden_model.get_parameter(name))
```

## 🔍 调试技巧

1. **检查模型加载**：确认Qwen3-30B-A3B模型正确加载
2. **验证分布式化**：检查参数是否正确分片
3. **监控传输状态**：查看发送/接收完成状态

## 🚨 常见问题

- **HF_HOME**: 确保设置`/data/hf` 目录
- **端口冲突**: 确保端口39500-39502可用

## 🔧 环境变量

- `TRAINING_STRATEGY`: 训练策略 (`pure_mp`, `hybrid_dp_mp`)
- `INFERENCE_STRATEGY`: 推理策略 (`pure_mp`, `hybrid_dp_mp`, 默认: `pure_mp`)
- `HF_HOME`: HuggingFace缓存目录 (默认: `/data/hf`)