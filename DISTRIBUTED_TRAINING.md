# 分布式训练使用指南

## 单机八卡分布式训练

### 方法1：使用 torchrun (推荐)

```bash
# 使用8个GPU
./train_distributed.sh 8

# 或使用4个GPU  
./train_distributed.sh 4
```

### 方法2：使用自定义启动脚本

```bash
# 使用8个GPU
python run_distributed_training.py --gpus 8

# 使用4个GPU
python run_distributed_training.py --gpus 4
```

### 方法3：直接使用 torchrun

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 train_model.py
```

## 环境要求

1. **PyTorch**: 版本 >= 1.9.0，支持分布式训练
2. **CUDA**: 确保所有GPU可见
3. **NCCL**: 后端通信库，PyTorch通常会自动包含

## 检查GPU可用性

```bash
# 检查GPU数量
nvidia-smi --list-gpus

# 检查PyTorch是否能识别GPU
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
```

## 性能优化建议

### 批量大小设置
- **单GPU批量大小**: 8 (配置中的batch_size)
- **总批量大小**: 8 × GPU数量
- 例如8卡训练时，总批量大小为64

### 数据加载器配置
- `dataloader_num_workers = 4`: 每个GPU 4个数据加载进程
- `pin_memory = True`: 加速GPU数据传输

### 混合精度训练
- `fp16 = True`: 节省显存，加速训练
- 自动损失缩放，防止梯度underflow

## 监控训练进展

### 只有主进程(rank=0)会显示输出：
- 训练进度条
- 损失和准确率
- 模型保存信息

### 其他进程(rank≠0)：
- 静默运行
- 参与梯度同步
- 不显示训练日志

## 故障排除

### 1. 端口冲突
如果遇到端口占用错误：
```bash
# 查看端口使用
netstat -tulpn | grep :29500

# 或者使用自定义端口
torchrun --master_port 29501 --standalone --nnodes=1 --nproc_per_node=8 train_model.py
```

### 2. NCCL通信错误
```bash
# 设置环境变量调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### 3. 显存不足
- 减小batch_size (例如改为4或2)
- 启用梯度检查点：在模型中添加`gradient_checkpointing=True`

### 4. 数据加载慢
- 减少`dataloader_num_workers`
- 确保数据文件在快速存储设备上(如SSD)

## 与单卡训练的区别

| 特性 | 单卡训练 | 分布式训练 |
|------|----------|------------|
| 启动方式 | `python train_model.py` | `./train_distributed.sh` |
| 总批量大小 | batch_size | batch_size × GPU数量 |
| 训练速度 | 1x | ~GPU数量×0.8x (通信开销) |
| 显存使用 | 单GPU显存 | 每个GPU独立显存 |
| 模型保存 | 直接保存 | 只在主进程保存 |

## 训练完成后

训练完成后，模型文件保存在 `./qwen_trajectory_model/best_model.pt`，可以直接用于推理，无需特殊处理。

## 性能预期

在8卡A100/H100环境下：
- **加速比**: 约6-7倍 (考虑通信开销)
- **显存使用**: 每卡独立，不共享
- **训练时间**: 相比单卡减少约85%