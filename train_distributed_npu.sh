#!/bin/bash

# 昇腾NPU分布式训练启动脚本
# 适用于单机8卡昇腾910B训练

echo "=== 昇腾NPU分布式训练启动脚本 ==="

# 检查昇腾NPU环境
if ! command -v npu-smi &> /dev/null; then
    echo "错误: 未找到npu-smi命令，请确认昇腾NPU环境已正确安装"
    exit 1
fi

# 检查可用的NPU数量
NPU_COUNT=$(npu-smi info | grep "NPU" | wc -l)
echo "检测到 ${NPU_COUNT} 个NPU设备"

if [ ${NPU_COUNT} -lt 1 ]; then
    echo "错误: 没有检测到可用的NPU设备"
    exit 1
fi

# 设置要使用的NPU数量 (默认为8，可以根据实际情况调整)
NNPUS=${1:-8}

if [ ${NNPUS} -gt ${NPU_COUNT} ]; then
    echo "警告: 请求使用 ${NNPUS} 个NPU，但只检测到 ${NPU_COUNT} 个NPU"
    echo "将使用所有可用的 ${NPU_COUNT} 个NPU"
    NNPUS=${NPU_COUNT}
fi

echo "使用 ${NNPUS} 个NPU进行分布式训练"

# 设置昇腾NPU环境变量
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I | awk '{print $1}')

# 检查Python环境中是否有torch_npu
if ! python3 -c "import torch_npu" 2>/dev/null; then
    echo "错误: 未找到torch_npu模块，请确认昇腾PyTorch插件已正确安装"
    echo "安装命令: pip install torch_npu"
    exit 1
fi

echo "昇腾NPU环境检查通过"

# 使用 torchrun 启动分布式训练
echo "启动训练..."
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NNPUS} \
    train_model_npu.py

echo "NPU分布式训练完成"