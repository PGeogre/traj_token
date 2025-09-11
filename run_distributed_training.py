#!/usr/bin/env python3
"""
分布式训练启动脚本
支持单机多卡训练
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

def find_free_port():
    """找到一个空闲的端口"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def main():
    parser = argparse.ArgumentParser(description='分布式训练启动脚本')
    parser.add_argument('--gpus', type=int, default=8, help='使用的GPU数量')
    parser.add_argument('--master_port', type=int, default=None, help='主进程端口')
    parser.add_argument('--node_rank', type=int, default=0, help='节点rank')
    parser.add_argument('--nnodes', type=int, default=1, help='节点数量')
    parser.add_argument('--master_addr', type=str, default='localhost', help='主进程地址')
    
    args = parser.parse_args()
    
    # 自动找到空闲端口
    if args.master_port is None:
        args.master_port = find_free_port()
    
    # 设置环境变量
    env = os.environ.copy()
    env['MASTER_ADDR'] = args.master_addr
    env['MASTER_PORT'] = str(args.master_port)
    env['WORLD_SIZE'] = str(args.gpus * args.nnodes)
    
    print(f"启动分布式训练:")
    print(f"  - GPU数量: {args.gpus}")
    print(f"  - 主进程地址: {args.master_addr}:{args.master_port}")
    print(f"  - 总进程数: {env['WORLD_SIZE']}")
    
    # 启动多个进程
    processes = []
    
    try:
        for local_rank in range(args.gpus):
            # 计算全局rank
            global_rank = args.node_rank * args.gpus + local_rank
            
            # 设置进程特定的环境变量
            process_env = env.copy()
            process_env['RANK'] = str(global_rank)
            process_env['LOCAL_RANK'] = str(local_rank)
            process_env['CUDA_VISIBLE_DEVICES'] = str(local_rank)
            
            # 启动训练进程
            cmd = [sys.executable, 'train_model.py']
            
            print(f"启动进程 {global_rank} (本地rank: {local_rank})...")
            
            process = subprocess.Popen(
                cmd,
                env=process_env,
                stdout=subprocess.PIPE if local_rank != 0 else None,
                stderr=subprocess.PIPE if local_rank != 0 else None,
            )
            processes.append(process)
        
        # 等待所有进程完成
        for i, process in enumerate(processes):
            return_code = process.wait()
            if return_code != 0:
                print(f"进程 {i} 异常退出，返回码: {return_code}")
                # 读取错误输出
                if process.stderr:
                    stderr_output = process.stderr.read().decode()
                    print(f"进程 {i} 错误输出:\n{stderr_output}")
                # 终止其他进程
                for other_process in processes:
                    if other_process != process and other_process.poll() is None:
                        other_process.terminate()
                sys.exit(return_code)
        
        print("所有训练进程成功完成!")
        
    except KeyboardInterrupt:
        print("\n收到中断信号，正在终止所有进程...")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        print("所有进程已终止")

if __name__ == "__main__":
    main()