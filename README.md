# 轨迹分类模型训练项目

基于Qwen2.5-0.5B的船舶轨迹分类模型，支持预训练+微调的完整训练流程。

## 项目结构

```
token_for_nj/
├── README.md                           # 项目说明文档
├── run_training.sh                     # 训练脚本（主入口）
├── train_dataset_demo.jsonl           # 原始训练数据
├── train_dataset_demo_pretrain.jsonl  # 生成的预训练数据（自动生成）
├── pretrain_config.py                 # 预训练和微调配置
├── train_pretrain.py                  # 预训练脚本
├── train_finetune.py                  # 微调脚本
├── pretrain_data_processor.py         # 预训练数据处理
├── pretrain_models.py                 # 预训练模型定义
├── data_loader.py                     # 数据加载器
└── inference.py                       # 推理脚本
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- transformers 4.40+
- CUDA支持的GPU（推荐）

```bash
pip install torch transformers scikit-learn tqdm numpy
```

## 数据准备

### 数据格式

训练数据应为JSONL格式，每行包含一个样本：

```json
{"text": "YEAR_2021 MONTH_9 DAY_17 HOUR_6 MINUTE_11 SECOND_42 H3_CHAR_0_8 H3_CHAR_1_a ... SPD_STOP COG_NE Bulk_Carrier POINT_END", "label": 0, "source_file": "104694.csv"}
```

### 数据字段说明

- `text`: 船舶轨迹的token化文本序列
  - 时间tokens: `YEAR_2021`, `MONTH_9`, `DAY_17`, `HOUR_6`, `MINUTE_11`, `SECOND_42`
  - 地理位置tokens: `H3_CHAR_0_8`, `H3_CHAR_1_a` 等
  - 速度tokens: `SPD_STOP`, `SPD_SLOW`, `SPD_MID`, `SPD_FAST`, `SPD_HIGH`
  - 航向tokens: `COG_N`, `COG_NE`, `COG_E` 等
  - 船舶类型tokens: `Bulk_Carrier`, `Oil_Tanker` 等
  - 结构tokens: `POINT_END`
- `label`: 分类标签（整数）
- `source_file`: 数据来源文件（可选）

### 数据放置

将训练数据命名为 `train_dataset_demo.jsonl` 并放置在项目根目录。

## 模型配置

### 基础模型路径

在 `pretrain_config.py` 中配置基础模型路径：

```python
model_name = "/home/maomao/pretrained_model/qwen2.5-0.5b"  # 修改为你的模型路径
```

### 主要配置参数

```python
# 预训练配置
class PretrainConfig:
    batch_size = 4          # 批次大小
    learning_rate = 5e-5    # 学习率
    num_epochs = 5          # 训练轮数
    max_length = 512        # 最大序列长度
    mask_prob = 0.15        # MLM掩码概率

# 微调配置
class FineTuneConfig:
    batch_size = 4          # 批次大小
    learning_rate = 2e-5    # 学习率（通常比预训练小）
    num_epochs = 10         # 训练轮数
    num_labels = 14         # 分类类别数
```

## 训练流程

### 1. 仅预训练

```bash
./run_training.sh pretrain
```

### 2. 仅微调

```bash
./run_training.sh finetune /path/to/pretrained/model
```

### 3. 完整流程（预训练 + 微调）

```bash
./run_training.sh full
```

## 训练输出

### 模型保存位置

```
../model_out/
├── trajectory_pretrain_model/          # 预训练模型
│   ├── best_model/                     # 最佳检查点
│   ├── final_model/                    # 最终模型 ⭐
│   └── checkpoint-*/                   # 训练检查点
├── trajectory_finetune_model/          # 微调模型
│   ├── best_model/                     # 最佳检查点 ⭐
│   ├── final_model/                    # 最终模型
│   └── checkpoint-*/                   # 训练检查点
└── logs/                              # 训练日志
    ├── pretrain_*.log
    └── finetune_*.log
```

### 推荐使用的模型

- **预训练后微调使用**: `../model_out/trajectory_finetune_model/best_model/`
- **仅预训练使用**: `../model_out/trajectory_pretrain_model/final_model/`

## 推理使用

```python
from inference import TrajectoryInference

# 加载微调后的最终模型
inferencer = TrajectoryInference("../model_out/trajectory_finetune_model/best_model/")

# 预测单个样本
text = "YEAR_2021 MONTH_9 DAY_17 ... Bulk_Carrier POINT_END"
prediction = inferencer.predict(text)
print(f"预测类别: {prediction}")

# 批量预测
texts = [text1, text2, text3]
predictions = inferencer.predict_batch(texts)
```

## 分布式训练

项目自动检测GPU数量：
- **单GPU**: 自动使用单卡训练
- **多GPU**: 自动启用分布式训练

手动指定GPU：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_training.sh pretrain
```

## 训练监控

### 查看训练日志

```bash
# 实时查看预训练日志
tail -f ../model_out/logs/pretrain_*.log

# 实时查看微调日志  
tail -f ../model_out/logs/finetune_*.log
```

### 训练指标

- **预训练**: 关注MLM损失和掩码准确率
- **微调**: 关注分类准确率和验证损失

## 性能优化

### 提升训练速度

1. **增加批次大小**: 根据GPU显存调整 `batch_size`
2. **梯度累积**: 设置 `gradient_accumulation_steps`
3. **混合精度**: 启用 `fp16 = True`（默认开启）

### 提升模型性能

1. **调整学习率**: 预训练用较大学习率，微调用较小学习率
2. **增加训练轮数**: 根据验证集表现调整 `num_epochs`
3. **早停机制**: 防止过拟合（已默认启用）

## 常见问题

### Q: 预训练数据每次都重新生成？
A: 现已修复，预训练数据生成后会自动复用，除非手动删除 `train_dataset_demo_pretrain.jsonl`。

### Q: 如何修改分类类别数？
A: 在 `pretrain_config.py` 中修改 `FineTuneConfig.num_labels`。

### Q: 训练显存不够？
A: 减少 `batch_size`，或启用梯度累积 `gradient_accumulation_steps`。

### Q: 如何恢复中断的训练？
A: 训练会自动从最新检查点恢复，检查点保存在 `checkpoint-*` 目录中。

## 技术特点

- ✅ **智能数据复用**: 预训练数据生成一次，多次使用
- ✅ **自动分布式**: 多GPU环境自动启用分布式训练
- ✅ **领域适应**: 专门的MLM掩码策略适应轨迹数据
- ✅ **完整流程**: 支持预训练→微调的完整pipeline
- ✅ **监控友好**: 详细的训练日志和进度显示
- ✅ **容错设计**: 异常处理和训练恢复机制

## License

MIT License