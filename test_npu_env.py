#!/usr/bin/env python3
"""
昇腾NPU训练环境快速测试
验证模型是否能在NPU上正常运行
"""

import torch
import torch.nn as nn
import numpy as np

def test_npu_basic():
    """测试基本NPU功能"""
    print("=== NPU基础功能测试 ===")
    
    try:
        import torch_npu
        print("✓ torch_npu模块导入成功")
    except ImportError:
        print("✗ torch_npu模块导入失败")
        return False
    
    # 检查NPU可用性
    if not torch.npu.is_available():
        print("✗ NPU不可用")
        return False
    
    npu_count = torch.npu.device_count()
    print(f"✓ 检测到 {npu_count} 个NPU设备")
    
    # 测试NPU设备信息
    for i in range(npu_count):
        device_name = torch.npu.get_device_name(i)
        print(f"  NPU {i}: {device_name}")
    
    return True

def test_npu_operations():
    """测试NPU张量操作"""
    print("\n=== NPU张量操作测试 ===")
    
    try:
        # 创建NPU张量
        device = torch.device('npu:0')
        x = torch.randn(3, 4, device=device)
        y = torch.randn(4, 5, device=device)
        
        print(f"✓ NPU张量创建成功")
        print(f"  x shape: {x.shape}, device: {x.device}")
        print(f"  y shape: {y.shape}, device: {y.device}")
        
        # 矩阵乘法
        z = torch.mm(x, y)
        print(f"✓ NPU矩阵乘法成功: {z.shape}")
        
        # 测试梯度计算
        x.requires_grad_(True)
        loss = z.sum()
        loss.backward()
        print(f"✓ NPU梯度计算成功")
        
        return True
        
    except Exception as e:
        print(f"✗ NPU张量操作失败: {e}")
        return False

def test_npu_model():
    """测试NPU模型训练"""
    print("\n=== NPU模型训练测试 ===")
    
    try:
        device = torch.device('npu:0')
        
        # 创建简单模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
                
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel().to(device)
        print("✓ 模型创建并移至NPU成功")
        
        # 创建优化器
        try:
            optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=0.01)
            print("✓ NPU优化器创建成功")
        except:
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            print("✓ 标准优化器创建成功")
        
        # 测试训练步骤
        x = torch.randn(32, 10, device=device)
        y = torch.randn(32, 1, device=device)
        
        # 前向传播
        pred = model(x)
        loss = nn.MSELoss()(pred, y)
        print(f"✓ 前向传播成功，损失: {loss.item():.4f}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("✓ 反向传播和参数更新成功")
        
        # 测试混合精度
        try:
            model_fp16 = model.half()
            x_fp16 = x.half()
            pred_fp16 = model_fp16(x_fp16)
            print("✓ 混合精度推理成功")
        except Exception as e:
            print(f"⚠ 混合精度测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ NPU模型训练测试失败: {e}")
        return False

def test_npu_distributed():
    """测试NPU分布式通信"""
    print("\n=== NPU分布式通信测试 ===")
    
    try:
        import torch.distributed as dist
        
        # 检查HCCL后端支持
        if hasattr(dist.Backend, 'HCCL'):
            print("✓ HCCL后端支持检测成功")
        else:
            print("⚠ HCCL后端未找到，可能使用NCCL")
        
        # 这里只是检测，不实际初始化分布式
        # 因为需要多进程环境
        print("✓ 分布式模块导入成功")
        print("  注意: 完整的分布式测试需要多进程环境")
        
        return True
        
    except Exception as e:
        print(f"✗ 分布式通信测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("昇腾NPU训练环境快速测试")
    print("=" * 50)
    
    tests = [
        ("基础功能", test_npu_basic),
        ("张量操作", test_npu_operations),
        ("模型训练", test_npu_model),
        ("分布式支持", test_npu_distributed),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    
    passed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！NPU环境已就绪，可以开始训练")
        return 0
    else:
        print("\n⚠ 部分测试失败，请检查NPU环境配置")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())