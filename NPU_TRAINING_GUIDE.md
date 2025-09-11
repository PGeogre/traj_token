# 昇腾910B NPU分布式训练指南

## 环境准备

### 1. 硬件要求
- 单机8卡昇腾910B NPU
- 充足的内存和存储空间

### 2. 软件环境安装

#### CANN工具包安装
```bash
# 下载CANN工具包 (版本7.0+)
# 官网: https://www.hiascend.com/software/cann

# 安装CANN
chmod +x Ascend-cann-toolkit_7.0.0_linux-aarch64.run
./Ascend-cann-toolkit_7.0.0_linux-aarch64.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### PyTorch NPU插件安装
```bash
# 安装PyTorch (建议版本1.11+)
pip install torch torchvision

# 安装昇腾PyTorch插件
pip install torch_npu

# 或从源码安装最新版本
git clone https://gitee.com/ascend/pytorch.git
cd pytorch
pip install -r requirements.txt
python setup.py install
```

## 环境检测

### 自动检测环境
```bash
# 运行环境检测脚本
./check_npu_env.sh
```

### 手动检测
```bash
# 检查NPU设备
npu-smi info

# 检查Python环境
python3 -c "
import torch
import torch_npu
print('NPU可用:', torch.npu.is_available())
print('NPU数量:', torch.npu.device_count())
"
```

## 训练启动

### 方法1: 使用启动脚本 (推荐)
```bash
# 使用8个NPU
./train_distributed_npu.sh 8

# 使用4个NPU
./train_distributed_npu.sh 4
```

### 方法2: 使用Python启动器
```bash
# 使用8个NPU
python launch_npu_training.py --npus 8

# 使用4个NPU
python launch_npu_training.py --npus 4
```

### 方法3: 直接使用torchrun
```bash
# 设置环境变量
source setup_npu_env.sh

# 启动训练
torchrun --standalone --nnodes=1 --nproc_per_node=8 train_model_npu.py
```

## 配置说明

### NPU专用配置 (train_config_npu.py)

| 参数 | 值 | 说明 |
|------|-----|------|
| `batch_size` | 6 | 每NPU批量大小，8卡总共48 |
| `learning_rate` | 1e-5 | NPU推荐较小学习率 |
| `max_length` | 512 | 序列长度，避免显存不足 |
| `fp16` | True | 混合精度训练，节省显存 |
| `dataloader_num_workers` | 2 | 数据加载进程数 |

### 与GPU训练的差异

| 特性 | GPU (CUDA) | NPU (昇腾) |
|------|------------|------------|
| 后端 | NCCL | HCCL |
| 设备 | `cuda:0` | `npu:0` |
| 优化器 | AdamW | NpuFusedAdamW |
| 混合精度 | AMP | 原生半精度 |
| 分布式初始化 | `nccl` | `hccl` |

## 性能优化建议

### 1. 内存优化
```python
# 使用混合精度
model = model.half()

# 梯度检查点 (如果显存不足)
model.gradient_checkpointing_enable()
```

### 2. 数据加载优化
```python
# 减少数据加载进程
dataloader_num_workers = 2

# 启用内存锁定
pin_memory = True
```

### 3. 批量大小调整
- 开始时使用较小的batch_size (如4)
- 根据显存使用情况逐渐增加
- 8卡建议总batch_size在32-64之间

## 故障排除

### 1. 环境问题
```bash
# CANN环境未设置
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# torch_npu导入失败
pip install torch_npu

# NPU设备不可见
npu-smi info
```

### 2. 训练问题
```bash
# HCCL通信错误
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I | awk '{print $1}')

# 显存不足
# 减小batch_size或max_length
```

### 3. 性能问题
```bash
# 数据加载慢
# 减少dataloader_num_workers
# 使用更快的存储设备

# 训练速度慢
# 检查NPU利用率: npu-smi info
# 启用混合精度训练
```

## 监控和调试

### NPU状态监控
```bash
# 实时监控NPU使用情况
watch -n 1 npu-smi info

# 查看详细信息
npu-smi info -t board -c -d
```

### 训练日志
- 只有主进程 (rank=0) 显示训练日志
- 模型自动保存在 `./qwen_trajectory_model_npu/`
- 支持断点续训

## 性能基准

在8卡昇腾910B环境下的预期性能：

| 指标 | 预期值 |
|------|--------|
| 训练加速比 | 6-7倍 |
| 内存利用率 | 70-85% |
| NPU利用率 | 85-95% |
| 训练时间节省 | 85%+ |

## 技术支持

- **华为昇腾官网**: https://www.hiascend.com/
- **PyTorch NPU仓库**: https://gitee.com/ascend/pytorch
- **技术文档**: https://www.hiascend.com/document
- **社区论坛**: https://bbs.huaweicloud.com/forum/forum-726-1.html