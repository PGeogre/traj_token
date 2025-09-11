#!/usr/bin/env python3
"""
昇腾NPU分布式训练启动器
简化版本，自动处理环境设置
"""

import os
import subprocess
import sys
import argparse

def check_npu_environment():
    """检查NPU环境"""
    try:
        import torch_npu
        import torch
        if torch.npu.is_available():
            return True, torch.npu.device_count()
        else:
            return False, 0
    except ImportError:
        return False, 0

def main():
    parser = argparse.ArgumentParser(description='昇腾NPU分布式训练启动器')
    parser.add_argument('--npus', type=int, default=8, help='使用的NPU数量')
    parser.add_argument('--master_port', type=int, default=29500, help='主进程端口')
    
    args = parser.parse_args()
    
    print(f"=== 昇腾NPU分布式训练启动器 ===")
    
    # 检查NPU环境
    npu_available, npu_count = check_npu_environment()
    
    if not npu_available:
        print("错误: NPU环境不可用")
        print("请检查:")
        print("1. 昇腾CANN工具包是否已安装")
        print("2. torch_npu是否已安装: pip install torch_npu")
        print("3. NPU驱动是否正常")
        sys.exit(1)
    
    print(f"✓ 检测到 {npu_count} 个NPU设备")
    
    # 调整NPU使用数量
    if args.npus > npu_count:
        print(f"警告: 请求 {args.npus} 个NPU，但只有 {npu_count} 个可用")
        args.npus = npu_count
    
    print(f"使用 {args.npus} 个NPU进行训练")
    
    # 设置环境变量
    env = os.environ.copy()
    env['MASTER_ADDR'] = 'localhost'
    env['MASTER_PORT'] = str(args.master_port)
    env['WORLD_SIZE'] = str(args.npus)
    
    # 昇腾特有环境变量
    env['HCCL_WHITELIST_DISABLE'] = '1'
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        env['HCCL_IF_IP'] = local_ip
    except:
        env['HCCL_IF_IP'] = '127.0.0.1'
    
    print(f"环境变量设置完成")
    print(f"主进程地址: localhost:{args.master_port}")
    
    # 启动训练
    cmd = [
        sys.executable, '-m', 'torch.distributed.launch',
        '--standalone',
        '--nnodes=1',
        f'--nproc_per_node={args.npus}',
        'train_model_npu.py'
    ]
    
    print(f"启动命令: {' '.join(cmd)}")
    print("开始训练...")
    
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print("训练成功完成！")
    except subprocess.CalledProcessError as e:
        print(f"训练失败，返回码: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(1)

if __name__ == "__main__":
    main()