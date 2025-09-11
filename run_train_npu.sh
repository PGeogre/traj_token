#!/bin/bash

# 昇腾NPU轨迹预训练+微调完整训练脚本
# 适用于单机8卡昇腾NPU训练环境
# 使用方法：
# 1. 只预训练：./run_train_npu.sh pretrain
# 2. 只微调：./run_train_npu.sh finetune /path/to/pretrained/model
# 3. 完整流程（预训练+微调）：./run_train_npu.sh full

set -e  # 遇到错误立即退出

echo "=== 昇腾NPU轨迹模型训练脚本 ==="

# 配置参数
BASE_DIR="/home/maomao/project/token_for_nj"
OUTPUT_BASE="../model_out"
PRETRAIN_OUTPUT="$OUTPUT_BASE/trajectory_pretrain_model_npu"
FINETUNE_OUTPUT="$OUTPUT_BASE/trajectory_finetune_model_npu"
LOG_DIR="$OUTPUT_BASE/logs_npu"

# 创建必要的目录
mkdir -p "$LOG_DIR"

# 获取当前时间戳
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# 检查昇腾NPU环境
function check_npu_environment() {
    echo "检查昇腾NPU环境..."
    
    # 检查npu-smi命令
    if ! command -v npu-smi &> /dev/null; then
        echo "❌ 错误: 未找到npu-smi命令"
        echo "请确认昇腾NPU驱动和工具已正确安装"
        exit 1
    fi
    
    # 检查NPU设备数量
    NPU_COUNT=$(npu-smi info 2>/dev/null | grep "NPU" | wc -l || echo "0")
    echo "检测到 ${NPU_COUNT} 个NPU设备"
    
    if [ ${NPU_COUNT} -lt 1 ]; then
        echo "❌ 错误: 没有检测到可用的NPU设备"
        echo "请检查NPU设备状态：npu-smi info"
        exit 1
    fi
    
    # 检查Python环境中的torch_npu
    if ! python3 -c "import torch_npu" 2>/dev/null; then
        echo "❌ 错误: 未找到torch_npu模块"
        echo "请安装昇腾PyTorch插件："
        echo "  pip install torch_npu"
        exit 1
    fi
    
    # 检查其他必要的Python包
    if ! python3 -c "import torch, transformers, sklearn, tqdm" 2>/dev/null; then
        echo "❌ 错误: 缺少必要的Python包"
        echo "请安装所需依赖："
        echo "  pip install torch transformers scikit-learn tqdm"
        exit 1
    fi
    
    echo "✅ 昇腾NPU环境检查通过"
    echo "NPU设备数量: ${NPU_COUNT}"
    
    # 显示NPU设备信息
    echo "NPU设备详情:"
    npu-smi info | head -10
    
    return 0
}

# 设置昇腾NPU环境变量
function setup_npu_env() {
    echo "设置昇腾NPU环境变量..."
    
    # 昇腾NPU核心环境变量
    export HCCL_WHITELIST_DISABLE=1
    export HCCL_IF_IP=$(hostname -I | awk '{print $1}')
    
    # NPU优化相关环境变量
    export NPU_CALCULATE_DEVICE=0,1,2,3,4,5,6,7
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    
    # 内存和性能优化
    export HCCL_BUFFSIZE=120
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    
    # 日志级别（可选）
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    export ASCEND_GLOBAL_LOG_LEVEL=1
    
    echo "✅ NPU环境变量设置完成"
}

echo "时间: $(date)"
echo "基础目录: $BASE_DIR"
echo "日志目录: $LOG_DIR"

# 进入工作目录
cd "$BASE_DIR"

function run_pretraining_npu() {
    echo ""
    echo "========================="
    echo "开始NPU预训练阶段"
    echo "========================="
    
    local log_file="$LOG_DIR/pretrain_npu_$TIMESTAMP.log"
    echo "预训练日志: $log_file"
    echo "预训练模型保存路径: $PRETRAIN_OUTPUT"
    
    # 检查训练脚本是否存在
    if [ ! -f "train_pretrain.py" ]; then
        echo "❌ 错误: 未找到预训练脚本 train_pretrain.py"
        exit 1
    fi
    
    # 使用所有可用的NPU进行分布式训练
    local npu_count=${NPU_COUNT:-8}  # 默认使用8个NPU
    echo "使用 ${npu_count} 个NPU进行分布式预训练"
    
    # 使用torchrun启动分布式预训练
    echo "启动NPU分布式预训练..."
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=${npu_count} \
        train_pretrain.py \
        --config_file train_config_npu.py \
        --device_type npu \
        --output_dir "$PRETRAIN_OUTPUT" \
        2>&1 | tee "$log_file"
    
    # 检查预训练是否成功
    if [ -d "$PRETRAIN_OUTPUT/best_model" ] || [ -f "$PRETRAIN_OUTPUT/best_model.pt" ]; then
        echo "✅ NPU预训练成功完成"
        echo "预训练模型保存在: $PRETRAIN_OUTPUT"
        
        # 显示模型文件
        echo "模型文件列表:"
        ls -la "$PRETRAIN_OUTPUT"/ || ls -la "$PRETRAIN_OUTPUT/best_model/" 2>/dev/null || true
    else
        echo "❌ NPU预训练失败，未找到模型输出"
        echo "请检查日志文件: $log_file"
        exit 1
    fi
}

function run_finetuning_npu() {
    local pretrained_path="$1"
    
    echo ""
    echo "========================="
    echo "开始NPU微调阶段"
    echo "========================="
    
    # 检查预训练模型是否存在
    if [ ! -d "$pretrained_path" ] && [ ! -f "$pretrained_path" ]; then
        echo "❌ 错误: 预训练模型不存在: $pretrained_path"
        echo "请先运行预训练或提供正确的预训练模型路径"
        exit 1
    fi
    
    local log_file="$LOG_DIR/finetune_npu_$TIMESTAMP.log"
    echo "微调日志: $log_file"
    echo "预训练模型路径: $pretrained_path"
    echo "微调模型保存路径: $FINETUNE_OUTPUT"
    
    # 检查微调脚本是否存在，优先使用NPU版本
    local finetune_script=""
    if [ -f "train_finetune_npu.py" ]; then
        finetune_script="train_finetune_npu.py"
    elif [ -f "train_model_npu.py" ]; then
        finetune_script="train_model_npu.py"
    elif [ -f "train_finetune.py" ]; then
        finetune_script="train_finetune.py"
        echo "⚠️  警告: 使用通用微调脚本，可能需要手动适配NPU"
    else
        echo "❌ 错误: 未找到微调训练脚本"
        exit 1
    fi
    
    echo "使用训练脚本: $finetune_script"
    
    # 使用所有可用的NPU进行分布式微调
    local npu_count=${NPU_COUNT:-8}
    echo "使用 ${npu_count} 个NPU进行分布式微调"
    
    # 启动NPU分布式微调
    echo "启动NPU分布式微调..."
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=${npu_count} \
        "$finetune_script" \
        --pretrained_model_path "$pretrained_path" \
        --config_file train_config_npu.py \
        --device_type npu \
        --output_dir "$FINETUNE_OUTPUT" \
        2>&1 | tee "$log_file"
    
    # 检查微调是否成功
    if [ -d "$FINETUNE_OUTPUT/best_model" ] || [ -f "$FINETUNE_OUTPUT/best_model.pt" ]; then
        echo "✅ NPU微调成功完成"
        echo "最终模型保存在: $FINETUNE_OUTPUT"
        
        # 显示模型文件
        echo "模型文件列表:"
        ls -la "$FINETUNE_OUTPUT"/ || ls -la "$FINETUNE_OUTPUT/best_model/" 2>/dev/null || true
        
        # 如果有验证报告，显示关键指标
        local report_file=""
        if [ -f "$FINETUNE_OUTPUT/best_model/validation_report.json" ]; then
            report_file="$FINETUNE_OUTPUT/best_model/validation_report.json"
        elif [ -f "$FINETUNE_OUTPUT/validation_report.json" ]; then
            report_file="$FINETUNE_OUTPUT/validation_report.json"
        fi
        
        if [ -n "$report_file" ]; then
            echo ""
            echo "验证结果摘要:"
            python3 -c "
import json
import sys
try:
    with open('$report_file') as f:
        report = json.load(f)
        print(f\"整体准确率: {report.get('accuracy', 'N/A'):.4f}\")
        print(f\"宏平均F1: {report.get('macro avg', {}).get('f1-score', 'N/A'):.4f}\")
        print(f\"加权平均F1: {report.get('weighted avg', {}).get('f1-score', 'N/A'):.4f}\")
except Exception as e:
    print(f\"无法读取验证报告: {e}\")
"
        fi
    else
        echo "❌ NPU微调失败，未找到模型输出"
        echo "请检查日志文件: $log_file"
        exit 1
    fi
}

function run_distributed_training_npu() {
    local mode="$1"
    local pretrained_path="$2"
    
    echo "使用现有的NPU分布式训练脚本..."
    
    if [ -f "train_distributed_npu.sh" ]; then
        echo "发现专用的NPU分布式训练脚本"
        chmod +x train_distributed_npu.sh
        
        local log_file="$LOG_DIR/distributed_npu_$TIMESTAMP.log"
        
        case "$mode" in
            "pretrain")
                echo "使用NPU分布式脚本进行预训练"
                ./train_distributed_npu.sh 2>&1 | tee "$log_file"
                ;;
            "finetune")
                echo "使用NPU分布式脚本进行微调"
                # 可能需要修改train_distributed_npu.sh来支持微调参数
                ./train_distributed_npu.sh 2>&1 | tee "$log_file"
                ;;
        esac
    else
        echo "未找到专用NPU分布式脚本，使用内置函数"
        case "$mode" in
            "pretrain")
                run_pretraining_npu
                ;;
            "finetune")
                run_finetuning_npu "$pretrained_path"
                ;;
        esac
    fi
}

function show_usage() {
    echo "用法: $0 <mode> [pretrained_model_path]"
    echo ""
    echo "模式:"
    echo "  pretrain              - 只运行NPU预训练"
    echo "  finetune <model_path> - 只运行NPU微调，需要指定预训练模型路径"
    echo "  full                  - 运行完整NPU流程（预训练 + 微调）"
    echo "  check                 - 只检查NPU环境"
    echo ""
    echo "示例:"
    echo "  $0 check"
    echo "  $0 pretrain"
    echo "  $0 finetune ../model_out/trajectory_pretrain_model_npu/best_model"
    echo "  $0 full"
}

# 主逻辑
case "$1" in
    "check")
        check_npu_environment
        setup_npu_env
        echo ""
        echo "✅ NPU环境检查完成，可以开始训练"
        ;;
    
    "pretrain")
        check_npu_environment
        setup_npu_env
        run_pretraining_npu
        echo ""
        echo "🎉 NPU预训练完成！"
        echo "接下来可以运行微调: $0 finetune $PRETRAIN_OUTPUT/best_model"
        ;;
    
    "finetune")
        if [ -z "$2" ]; then
            echo "❌ 错误: 微调模式需要指定预训练模型路径"
            show_usage
            exit 1
        fi
        check_npu_environment
        setup_npu_env
        run_finetuning_npu "$2"
        echo ""
        echo "🎉 NPU微调完成！"
        echo "最终模型可用于推理: $FINETUNE_OUTPUT/best_model"
        ;;
    
    "full")
        echo "运行完整NPU训练流程：预训练 -> 微调"
        check_npu_environment
        setup_npu_env
        
        # 1. 预训练
        run_pretraining_npu
        
        # 2. 微调
        local pretrain_model_path="$PRETRAIN_OUTPUT/best_model"
        if [ ! -d "$pretrain_model_path" ] && [ -f "$PRETRAIN_OUTPUT/best_model.pt" ]; then
            pretrain_model_path="$PRETRAIN_OUTPUT/best_model.pt"
        fi
        
        run_finetuning_npu "$pretrain_model_path"
        
        echo ""
        echo "🎉🎉🎉 完整NPU训练流程完成！"
        echo "预训练模型: $PRETRAIN_OUTPUT"
        echo "最终微调模型: $FINETUNE_OUTPUT"
        ;;
    
    "")
        echo "❌ 错误: 请指定训练模式"
        show_usage
        exit 1
        ;;
    
    *)
        echo "❌ 错误: 无效的模式 '$1'"
        show_usage
        exit 1
        ;;
esac

echo ""
echo "训练完成时间: $(date)"
echo "所有日志保存在: $LOG_DIR/"

# 显示磁盘使用情况
echo ""
echo "NPU模型文件磁盘使用情况:"
if [ -d "$OUTPUT_BASE" ]; then
    du -sh "$OUTPUT_BASE"/*npu* 2>/dev/null || echo "暂无NPU模型文件"
fi

echo ""
echo "NPU训练脚本执行完成 ✅"