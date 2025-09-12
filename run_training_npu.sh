#!/bin/bash

# 华为昇腾NPU轨迹预训练+微调完整训练脚本
# 使用方法：
# 1. 只预训练：./run_training_npu.sh pretrain
# 2. 只微调：./run_training_npu.sh finetune /path/to/pretrained/model
# 3. 完整流程（预训练+微调）：./run_training_npu.sh full

set -e  # 遇到错误立即退出

# 配置参数
BASE_DIR="/home/maomao/project/token_for_nj"
OUTPUT_BASE="../model_out"
PRETRAIN_OUTPUT="$OUTPUT_BASE/trajectory_pretrain_model_npu"
FINETUNE_OUTPUT="$OUTPUT_BASE/trajectory_finetune_model_npu"
LOG_DIR="$OUTPUT_BASE/logs"

# 创建必要的目录
mkdir -p "$LOG_DIR"

# 获取当前时间戳
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# 检查NPU环境
check_npu_environment() {
    echo "检查NPU环境..."
    
    # 检查npu-smi命令是否可用
    if ! command -v npu-smi &> /dev/null; then
        echo "错误: npu-smi 命令不可用，请检查昇腾驱动是否正确安装"
        exit 1
    fi
    
    # 检查NPU设备
    NPU_COUNT=$(npu-smi info | grep "NPU" | wc -l)
    if [ "$NPU_COUNT" -eq 0 ]; then
        echo "错误: 未检测到NPU设备"
        exit 1
    fi
    
    echo "检测到 $NPU_COUNT 个NPU设备"
    npu-smi info
    
    # 检查Python环境和NPU相关包
    if ! python -c "import torch, torch_npu, transformers" > /dev/null 2>&1; then
        echo "错误: 缺少必要的Python包 (torch, torch_npu, transformers)"
        echo "请先安装NPU版本的PyTorch和相关依赖"
        exit 1
    fi
    
    echo "NPU环境检查通过"
}

# 设置NPU环境变量
setup_npu_env() {
    # 设置NPU相关环境变量
    export ASCEND_RT_PATH="/usr/local/Ascend/nnae/latest"
    export PATH="/usr/local/Ascend/nnae/latest/bin:/usr/local/Ascend/nnae/latest/compiler/bin:$PATH"
    export PYTHONPATH="/usr/local/Ascend/nnae/latest/python/site-packages:$PYTHONPATH"
    export LD_LIBRARY_PATH="/usr/local/Ascend/nnae/latest/lib:$LD_LIBRARY_PATH"
    export ASCEND_OPP_PATH="/usr/local/Ascend/nnae/latest/opp"
    export ASCEND_AICPU_PATH="/usr/local/Ascend/nnae/latest"
    export ASCEND_TENSOR_COMPILER_INCLUDE="/usr/local/Ascend/nnae/latest/include"
    
    # 设置HCCL相关环境变量
    export HCCL_CONNECT_TIMEOUT=1800
    export HCCL_EXEC_TIMEOUT=1800
    
    # 优化NPU性能
    export TASK_QUEUE_ENABLE=1
    export PTCOPY_ENABLE=1
    export COMBINED_ENABLE=1
    export ACL_DUMP_DATA=0  # 生产环境关闭dump
    
    echo "NPU环境变量设置完成"
}

echo "=== 华为昇腾NPU轨迹模型训练脚本 ==="
echo "时间: $(date)"
echo "基础目录: $BASE_DIR"
echo "日志目录: $LOG_DIR"

# 进入工作目录
cd "$BASE_DIR"

# 检查并设置NPU环境
check_npu_environment
setup_npu_env

function run_pretraining_npu() {
    echo ""
    echo "==================="
    echo "开始NPU预训练阶段"
    echo "==================="
    
    local log_file="$LOG_DIR/pretrain_npu_$TIMESTAMP.log"
    echo "NPU预训练日志: $log_file"
    echo "NPU预训练模型保存路径: $PRETRAIN_OUTPUT"
    
    # 检查NPU数量决定是否使用分布式训练
    NPU_COUNT=$(npu-smi info | grep "NPU" | wc -l)
    if [ "$NPU_COUNT" -gt 1 ]; then
        echo "检测到 $NPU_COUNT 个NPU，使用分布式训练"
        
        # NPU分布式预训练
        python -m torch.distributed.launch \
            --nproc_per_node=$NPU_COUNT \
            --master_port=29500 \
            --use_env \
            train_pretrain_npu.py \
            2>&1 | tee "$log_file"
    else
        echo "使用单NPU训练"
        
        # 单NPU预训练
        python train_pretrain_npu.py \
            2>&1 | tee "$log_file"
    fi
    
    # 检查预训练是否成功
    if [ -d "$PRETRAIN_OUTPUT/final_model" ]; then
        echo "✅ NPU预训练成功完成"
        echo "预训练模型保存在: $PRETRAIN_OUTPUT/final_model"
        
        # 显示模型文件
        echo "模型文件列表:"
        ls -la "$PRETRAIN_OUTPUT/final_model/"
        
        # 显示NPU设备状态
        echo "训练完成后NPU状态:"
        npu-smi info -t power -i 0
    else
        echo "❌ NPU预训练失败，未找到模型输出"
        exit 1
    fi
}

function run_finetuning_npu() {
    local pretrained_path="$1"
    
    echo ""
    echo "==================="
    echo "开始NPU微调阶段"
    echo "==================="
    
    # 检查预训练模型是否存在
    if [ ! -d "$pretrained_path" ]; then
        echo "错误: 预训练模型不存在: $pretrained_path"
        echo "请先运行NPU预训练或提供正确的预训练模型路径"
        exit 1
    fi
    
    local log_file="$LOG_DIR/finetune_npu_$TIMESTAMP.log"
    echo "NPU微调日志: $log_file"
    echo "预训练模型路径: $pretrained_path"
    echo "NPU微调模型保存路径: $FINETUNE_OUTPUT"
    
    # 检查NPU数量决定是否使用分布式训练
    NPU_COUNT=$(npu-smi info | grep "NPU" | wc -l)
    if [ "$NPU_COUNT" -gt 1 ]; then
        echo "检测到 $NPU_COUNT 个NPU，使用分布式微调"
        
        # NPU分布式微调
        python -m torch.distributed.launch \
            --nproc_per_node=$NPU_COUNT \
            --master_port=29501 \
            --use_env \
            train_finetune_npu.py \
            --pretrained_model_path "$pretrained_path" \
            2>&1 | tee "$log_file"
    else
        echo "使用单NPU微调"
        
        # 单NPU微调
        python train_finetune_npu.py \
            --pretrained_model_path "$pretrained_path" \
            2>&1 | tee "$log_file"
    fi
    
    # 检查微调是否成功
    if [ -d "$FINETUNE_OUTPUT/best_model" ]; then
        echo "✅ NPU微调成功完成"
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
        
        # 显示NPU设备状态
        echo "训练完成后NPU状态:"
        npu-smi info -t power -i 0
    else
        echo "❌ NPU微调失败，未找到模型输出"
        exit 1
    fi
}

function show_usage() {
    echo "用法: $0 <mode> [pretrained_model_path]"
    echo ""
    echo "模式:"
    echo "  pretrain              - 只运行NPU预训练"
    echo "  finetune <model_path> - 只运行NPU微调，需要指定预训练模型路径"
    echo "  full                  - 运行完整流程（NPU预训练 + NPU微调）"
    echo ""
    echo "示例:"
    echo "  $0 pretrain"
    echo "  $0 finetune ../model_out/trajectory_pretrain_model_npu/best_model"
    echo "  $0 full"
    echo ""
    echo "NPU环境要求:"
    echo "  - 华为昇腾NPU设备"
    echo "  - CANN驱动和工具链"
    echo "  - torch_npu库"
}

function check_npu_resources() {
    echo ""
    echo "NPU资源使用情况:"
    npu-smi info
    echo ""
    echo "NPU内存使用情况:"
    npu-smi info -t usages -i 0
}

# 主逻辑
case "$1" in
    "pretrain")
        run_pretraining_npu
        echo ""
        echo "🎉 NPU预训练完成！"
        echo "接下来可以运行NPU微调: $0 finetune $PRETRAIN_OUTPUT/final_model"
        check_npu_resources
        ;;
    
    "finetune")
        if [ -z "$2" ]; then
            echo "错误: 微调模式需要指定预训练模型路径"
            show_usage
            exit 1
        fi
        run_finetuning_npu "$2"
        echo ""
        echo "🎉 NPU微调完成！"
        echo "最终模型可用于推理: $FINETUNE_OUTPUT/best_model"
        check_npu_resources
        ;;
    
    "full")
        echo "运行完整NPU训练流程：预训练 -> 微调"
        
        # 1. NPU预训练
        run_pretraining_npu
        
        # 等待一段时间让NPU设备冷却
        echo "等待NPU设备冷却..."
        sleep 30
        
        # 2. NPU微调
        run_finetuning_npu "$PRETRAIN_OUTPUT/final_model"
        
        echo ""
        echo "🎉🎉🎉 完整NPU训练流程完成！"
        echo "预训练模型: $PRETRAIN_OUTPUT/final_model"
        echo "最终微调模型: $FINETUNE_OUTPUT/best_model"
        check_npu_resources
        ;;
    
    "check")
        echo "检查NPU环境和资源状态"
        check_npu_environment
        check_npu_resources
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

# 清理NPU缓存
echo ""
echo "清理NPU缓存..."
python -c "
try:
    import torch_npu
    if torch_npu.npu.is_available():
        torch_npu.npu.empty_cache()
        print('NPU缓存清理完成')
    else:
        print('NPU不可用，跳过缓存清理')
except Exception as e:
    print(f'清理NPU缓存时出错: {e}')
"

echo "NPU训练脚本执行完成！"