#!/bin/bash

# 昇腾NPU环境检测和设置脚本

echo "=== 昇腾NPU环境检测脚本 ==="

# 1. 检查CANN工具包
echo "1. 检查CANN工具包..."
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    echo "  ✓ 发现CANN工具包安装路径: /usr/local/Ascend/ascend-toolkit/"
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f "$HOME/Ascend/ascend-toolkit/set_env.sh" ]; then
    echo "  ✓ 发现CANN工具包安装路径: $HOME/Ascend/ascend-toolkit/"
    source $HOME/Ascend/ascend-toolkit/set_env.sh
else
    echo "  ✗ 未找到CANN工具包，请先安装CANN"
    echo "    下载地址: https://www.hiascend.com/software/cann"
    exit 1
fi

# 2. 检查NPU设备
echo "2. 检查NPU设备..."
if command -v npu-smi &> /dev/null; then
    echo "  ✓ npu-smi命令可用"
    NPU_INFO=$(npu-smi info)
    NPU_COUNT=$(echo "$NPU_INFO" | grep "NPU" | wc -l)
    echo "  ✓ 检测到 ${NPU_COUNT} 个NPU设备"
    echo "NPU设备信息:"
    npu-smi info | grep -E "NPU|Health|Power"
else
    echo "  ✗ npu-smi命令不可用，请检查驱动安装"
    exit 1
fi

# 3. 检查Python环境
echo "3. 检查Python环境..."
PYTHON_VERSION=$(python3 --version 2>&1)
echo "  Python版本: $PYTHON_VERSION"

# 4. 检查PyTorch
echo "4. 检查PyTorch..."
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "  ✓ PyTorch版本: $TORCH_VERSION"
else
    echo "  ✗ PyTorch未安装"
    echo "  建议安装命令: pip install torch torchvision"
    exit 1
fi

# 5. 检查torch_npu
echo "5. 检查torch_npu..."
if python3 -c "import torch_npu" 2>/dev/null; then
    TORCH_NPU_VERSION=$(python3 -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null || echo "未知版本")
    echo "  ✓ torch_npu版本: $TORCH_NPU_VERSION"
    
    # 测试NPU可用性
    NPU_AVAILABLE=$(python3 -c "
import torch
import torch_npu
print('NPU可用:', torch.npu.is_available())
print('NPU数量:', torch.npu.device_count())
if torch.npu.is_available():
    for i in range(torch.npu.device_count()):
        print(f'NPU {i}:', torch.npu.get_device_name(i))
" 2>/dev/null)
    echo "$NPU_AVAILABLE"
else
    echo "  ✗ torch_npu未安装"
    echo "  安装命令:"
    echo "  pip install torch_npu"
    echo "  或参考官方文档: https://gitee.com/ascend/pytorch"
    exit 1
fi

# 6. 检查分布式训练支持
echo "6. 检查分布式训练支持..."
DIST_TEST=$(python3 -c "
import torch
import torch.distributed as dist
import torch_npu
print('分布式后端支持:')
print('  HCCL:', dist.is_available() and 'hccl' in dist.Backend._plugins)
print('  NCCL:', dist.is_available() and dist.is_nccl_available())
" 2>/dev/null)
echo "$DIST_TEST"

# 7. 环境变量建议
echo "7. 建议的环境变量设置:"
echo "export HCCL_WHITELIST_DISABLE=1"
echo "export HCCL_IF_IP=\$(hostname -I | awk '{print \$1}')"

# 8. 生成环境设置脚本
cat > setup_npu_env.sh << 'EOF'
#!/bin/bash
# 昇腾NPU环境变量设置

# CANN工具包环境
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f "$HOME/Ascend/ascend-toolkit/set_env.sh" ]; then
    source $HOME/Ascend/ascend-toolkit/set_env.sh
fi

# 分布式训练环境变量
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I | awk '{print $1}')

echo "昇腾NPU环境变量已设置"
EOF

chmod +x setup_npu_env.sh
echo "8. 已生成环境设置脚本: setup_npu_env.sh"

echo ""
echo "=== 环境检测完成 ==="
echo "使用方法:"
echo "1. 每次训练前运行: source setup_npu_env.sh"
echo "2. 启动分布式训练: ./train_distributed_npu.sh 8"