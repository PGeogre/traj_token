#!/bin/bash

# 轨迹预训练+微调完整训练脚本
# 使用方法：
# 1. 只预训练：./run_training.sh pretrain
# 2. 只微调：./run_training.sh finetune /path/to/pretrained/model
# 3. 完整流程（预训练+微调）：./run_training.sh full

set -e  # 遇到错误立即退出

# 配置参数
BASE_DIR="/home/maomao/project/token_for_nj"
OUTPUT_BASE="../model_out"
PRETRAIN_OUTPUT="$OUTPUT_BASE/trajectory_pretrain_model"
FINETUNE_OUTPUT="$OUTPUT_BASE/trajectory_finetune_model"
LOG_DIR="$OUTPUT_BASE/logs"

# 创建必要的目录
mkdir -p "$LOG_DIR"

# 获取当前时间戳
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# 检查CUDA环境
if ! nvidia-smi > /dev/null 2>&1; then
    echo "警告: 未检测到NVIDIA GPU，将使用CPU训练（速度会很慢）"
fi

# 检查Python环境
if ! python -c "import torch, transformers" > /dev/null 2>&1; then
    echo "错误: 缺少必要的Python包 (torch, transformers)"
    echo "请先安装: pip install torch transformers"
    exit 1
fi

echo "=== 轨迹模型训练脚本 ==="
echo "时间: $(date)"
echo "基础目录: $BASE_DIR"
echo "日志目录: $LOG_DIR"

# 进入工作目录
cd "$BASE_DIR"

function run_pretraining() {
    echo ""
    echo "==================="
    echo "开始预训练阶段"
    echo "==================="
    
    local log_file="$LOG_DIR/pretrain_$TIMESTAMP.log"
    echo "预训练日志: $log_file"
    
    echo "预训练模型保存路径: $PRETRAIN_OUTPUT"
    
    # 检查是否有多GPU可用
    local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null || echo "0")
    
    if [ "$gpu_count" -gt 1 ]; then
        echo "检测到 $gpu_count 个GPU，使用分布式训练"
        
        # 分布式预训练
        python -m torch.distributed.launch \
            --nproc_per_node=$gpu_count \
            --master_port=12345 \
            train_pretrain.py \
            2>&1 | tee "$log_file"
    else
        echo "使用单GPU/CPU训练"
        
        # 单卡预训练
        python train_pretrain.py \
            2>&1 | tee "$log_file"
    fi
    
    # 检查预训练是否成功
    if [ -d "$PRETRAIN_OUTPUT/best_model" ]; then
        echo "✅ 预训练成功完成"
        echo "预训练模型保存在: $PRETRAIN_OUTPUT/best_model"
        
        # 显示模型文件
        echo "模型文件列表:"
        ls -la "$PRETRAIN_OUTPUT/best_model/"
    else
        echo "❌ 预训练失败，未找到模型输出"
        exit 1
    fi
}

function run_finetuning() {
    local pretrained_path="$1"
    
    echo ""
    echo "==================="
    echo "开始微调阶段"
    echo "==================="
    
    # 检查预训练模型是否存在
    if [ ! -d "$pretrained_path" ]; then
        echo "错误: 预训练模型不存在: $pretrained_path"
        echo "请先运行预训练或提供正确的预训练模型路径"
        exit 1
    fi
    
    local log_file="$LOG_DIR/finetune_$TIMESTAMP.log"
    echo "微调日志: $log_file"
    echo "预训练模型路径: $pretrained_path"
    echo "微调模型保存路径: $FINETUNE_OUTPUT"
    
    # 检查是否有多GPU可用
    local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null || echo "0")
    
    if [ "$gpu_count" -gt 1 ]; then
        echo "检测到 $gpu_count 个GPU，使用分布式微调"
        
        # 分布式微调
        python -m torch.distributed.launch \
            --nproc_per_node=$gpu_count \
            --master_port=12346 \
            train_finetune.py \
            --pretrained_model_path "$pretrained_path" \
            2>&1 | tee "$log_file"
    else
        echo "使用单GPU/CPU微调"
        
        # 单卡微调
        python train_finetune.py \
            --pretrained_model_path "$pretrained_path" \
            2>&1 | tee "$log_file"
    fi
    
    # 检查微调是否成功
    if [ -d "$FINETUNE_OUTPUT/best_model" ]; then
        echo "✅ 微调成功完成"
        echo "最终模型保存在: $FINETUNE_OUTPUT/best_model"
        
        # 显示模型文件
        echo "模型文件列表:"
        ls -la "$FINETUNE_OUTPUT/best_model/"
        
        # 如果有验证报告，显示关键指标
        if [ -f "$FINETUNE_OUTPUT/best_model/validation_report.json" ]; then
            echo ""
            echo "验证结果摘要:"
            python -c "
import json
with open('$FINETUNE_OUTPUT/best_model/validation_report.json') as f:
    report = json.load(f)
    print(f\"整体准确率: {report.get('accuracy', 'N/A'):.4f}\")
    print(f\"宏平均F1: {report.get('macro avg', {}).get('f1-score', 'N/A'):.4f}\")
    print(f\"加权平均F1: {report.get('weighted avg', {}).get('f1-score', 'N/A'):.4f}\")
"
        fi
    else
        echo "❌ 微调失败，未找到模型输出"
        exit 1
    fi
}

function show_usage() {
    echo "用法: $0 <mode> [pretrained_model_path]"
    echo ""
    echo "模式:"
    echo "  pretrain              - 只运行预训练"
    echo "  finetune <model_path> - 只运行微调，需要指定预训练模型路径"
    echo "  full                  - 运行完整流程（预训练 + 微调）"
    echo ""
    echo "示例:"
    echo "  $0 pretrain"
    echo "  $0 finetune ../model_out/trajectory_pretrain_model/best_model"
    echo "  $0 full"
}

# 主逻辑
case "$1" in
    "pretrain")
        run_pretraining
        echo ""
        echo "🎉 预训练完成！"
        echo "接下来可以运行微调: $0 finetune $PRETRAIN_OUTPUT/best_model"
        ;;
    
    "finetune")
        if [ -z "$2" ]; then
            echo "错误: 微调模式需要指定预训练模型路径"
            show_usage
            exit 1
        fi
        run_finetuning "$2"
        echo ""
        echo "🎉 微调完成！"
        echo "最终模型可用于推理: $FINETUNE_OUTPUT/best_model"
        ;;
    
    "full")
        echo "运行完整训练流程：预训练 -> 微调"
        
        # 1. 预训练
        run_pretraining
        
        # 2. 微调
        run_finetuning "$PRETRAIN_OUTPUT/best_model"
        
        echo ""
        echo "🎉🎉🎉 完整训练流程完成！"
        echo "预训练模型: $PRETRAIN_OUTPUT/best_model"
        echo "最终微调模型: $FINETUNE_OUTPUT/best_model"
        ;;
    
    *)
        echo "错误: 无效的模式 '$1'"
        show_usage
        exit 1
        ;;
esac

echo ""
echo "训练完成时间: $(date)"
echo "所有日志保存在: $LOG_DIR/"

# 显示磁盘使用情况
echo ""
echo "模型文件磁盘使用情况:"
if [ -d "$OUTPUT_BASE" ]; then
    du -sh "$OUTPUT_BASE"/* 2>/dev/null || true
fi