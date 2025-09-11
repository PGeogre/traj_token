# 轨迹预训练+微调项目使用说明

本项目实现了轨迹数据的预训练+微调完整流程，用于船舶分类任务。

## 📁 项目结构

```
├── prepare_dataset.py          # 原始数据预处理
├── pretrain_data_processor.py  # 预训练数据处理（MLM掩码）
├── pretrain_models.py          # 预训练和微调模型架构
├── pretrain_config.py          # 配置文件
├── train_pretrain.py           # 预训练脚本
├── train_finetune.py           # 微调脚本
├── train_model.py              # 增强版原始训练脚本（支持预训练）
├── run_training.sh             # 一键运行脚本
└── data_loader.py              # 数据加载器
```

## 🚀 快速开始

### 方法1: 一键运行完整流程
```bash
# 运行完整的预训练+微调流程
./run_training.sh full
```

### 方法2: 分步运行

#### 步骤1: 预训练
```bash
# 只运行预训练
./run_training.sh pretrain

# 或者手动运行
python train_pretrain.py
```

#### 步骤2: 微调
```bash
# 使用预训练模型进行微调
./run_training.sh finetune ../model_out/trajectory_pretrain_model/best_model

# 或者手动运行
python train_finetune.py --pretrained_model_path ../model_out/trajectory_pretrain_model/best_model
```

## 🏗️ 技术架构

### 预训练阶段 (MLM)
- **目标**: 学习轨迹的空间-时间模式和船舶行为特征
- **任务**: Masked Language Modeling (MLM)
- **掩码策略**:
  - 40% 概率掩码H3地理位置tokens
  - 30% 概率掩码船舶类别tokens  
  - 20% 概率掩码速度/航向tokens
  - 10% 概率随机掩码其他tokens

### 微调阶段 (分类)
- **目标**: 船舶类型分类
- **策略**: 加载预训练backbone + 分类头
- **优化**: 分层学习率（backbone用较小LR）

## ⚙️ 关键配置

### 预训练配置 (pretrain_config.py)
```python
class PretrainConfig:
    mask_prob = 0.15          # 总体掩码概率
    batch_size = 8            # 预训练batch size
    learning_rate = 5e-5      # 预训练学习率
    num_epochs = 50           # 预训练轮数
    early_stopping_patience = 5
```

### 微调配置
```python
class FineTuneConfig:
    batch_size = 4            # 微调batch size
    learning_rate = 2e-5      # 微调学习率（更小）
    num_epochs = 20           # 微调轮数（更少）
    freeze_backbone_epochs = 0 # 冻结backbone轮数
```

## 📊 训练监控

训练过程中的关键指标：

### 预训练监控
- MLM Loss: 掩码预测损失
- MLM Accuracy: 掩码预测准确率
- 学习率变化

### 微调监控  
- 分类Loss和Accuracy
- 各类别的F1-score
- 混淆矩阵

## 🗂️ 输出文件

```
../model_out/
├── trajectory_pretrain_model/
│   ├── best_model/           # 最佳预训练模型
│   ├── final_model/          # 最终预训练模型  
│   └── training_log.jsonl    # 预训练日志
├── trajectory_finetune_model/
│   ├── best_model/           # 最佳微调模型（用于推理）
│   ├── final_model/          # 最终微调模型
│   ├── finetune_log.jsonl    # 微调日志
│   └── validation_report.json # 验证报告
└── logs/                     # 训练日志
```

## 🔧 高级用法

### 分布式训练
```bash
# 自动检测GPU数量并使用分布式训练
python -m torch.distributed.launch --nproc_per_node=4 train_pretrain.py
python -m torch.distributed.launch --nproc_per_node=4 train_finetune.py
```

### 自定义配置
```python
# 修改 pretrain_config.py 中的配置
config = PretrainConfig()
config.mask_prob = 0.20           # 增加掩码概率
config.h3_mask_prob = 0.5         # 更多H3掩码
config.learning_rate = 1e-4       # 调整学习率
```

### 使用预训练模型进行标准训练
```bash
# 使用增强版train_model.py
python train_model.py --use_pretrained --pretrained_model_path ../model_out/trajectory_pretrain_model/best_model
```

## 📈 预期效果

相比直接训练，预训练+微调应该能带来：

1. **更好的收敛**: 更快达到较高精度
2. **更强的泛化**: 在测试集上表现更稳定  
3. **数据效率**: 充分利用所有轨迹数据
4. **特征学习**: 更好理解轨迹的空间-时间模式

## 🐛 常见问题

### Q: 预训练数据生成失败
A: 检查原始数据路径，确保 `train_dataset_nj.jsonl` 存在

### Q: GPU内存不足
A: 减小 `batch_size` 或使用梯度累积

### Q: 分布式训练失败
A: 检查端口占用，可修改 `--master_port` 参数

### Q: 模型加载失败
A: 确保预训练模型路径正确，检查文件完整性

## 📝 日志分析

### 查看训练进度
```bash
# 实时查看预训练日志
tail -f ../model_out/logs/pretrain_*.log

# 查看微调日志
tail -f ../model_out/logs/finetune_*.log
```

### 解析训练指标
```python
import json

# 解析预训练日志
with open('../model_out/trajectory_pretrain_model/training_log.jsonl') as f:
    logs = [json.loads(line) for line in f]
    print(f"最佳验证损失: {min(log['val_loss'] for log in logs)}")

# 查看微调结果
with open('../model_out/trajectory_finetune_model/best_model/validation_report.json') as f:
    report = json.load(f)
    print(f"整体准确率: {report['accuracy']:.4f}")
```

---

🎯 **快速开始建议**: 直接运行 `./run_training.sh full` 体验完整流程！