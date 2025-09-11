#!/bin/bash

# 分布式训练启动脚本 - 使用 torchrun
# 适用于单机8卡训练

# 检查是否有足够的GPU
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 ${GPU_COUNT} 个GPU"

if [ ${GPU_COUNT} -lt 1 ]; then
    echo "错误: 没有检测到GPU"
    exit 1
fi

# 设置要使用的GPU数量 (默认为8，可以根据实际情况调整)
NGPUS=${1:-8}

if [ ${NGPUS} -gt ${GPU_COUNT} ]; then
    echo "警告: 请求使用 ${NGPUS} 个GPU，但只检测到 ${GPU_COUNT} 个GPU"
    echo "将使用所有可用的 ${GPU_COUNT} 个GPU"
    NGPUS=${GPU_COUNT}
fi

echo "使用 ${NGPUS} 个GPU进行分布式训练"

# 使用 torchrun 启动分布式训练
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NGPUS} \
    train_model.py

echo "分布式训练完成"