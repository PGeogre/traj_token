import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
import argparse

# 昇腾NPU相关导入
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    NPU_AVAILABLE = True
    print("昇腾NPU环境已加载")
except ImportError:
    NPU_AVAILABLE = False
    print("昇腾NPU环境未找到，将使用CPU/GPU")

from train_config_npu import TrainingConfigNPU
from data_loader import create_data_loaders

def setup_distributed_npu():
    """设置昇腾NPU分布式训练环境"""
    if not NPU_AVAILABLE:
        print("昇腾NPU不可用，使用CPU/GPU分布式训练")
        return setup_distributed_gpu()
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank % 8))
    else:
        print("未检测到分布式环境变量，将使用单NPU训练")
        return False, 0, 1, 0
    
    # 设置NPU设备
    torch.npu.set_device(local_rank)
    
    # 初始化进程组 - 昇腾使用HCCL后端
    dist.init_process_group(
        backend='hccl',  # 昇腾使用HCCL而不是NCCL
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return True, rank, world_size, local_rank

def setup_distributed_gpu():
    """GPU分布式训练环境（兜底方案）"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_id = int(os.environ['LOCAL_RANK'])
    else:
        print("未检测到分布式环境变量，将使用单卡训练")
        return False, 0, 1, 0
    
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return True, rank, world_size, gpu_id

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    """判断是否为主进程"""
    return rank == 0

class TrajectoryClassifier(nn.Module):
    def __init__(self, model_name, num_labels, tokenizer=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # 如果tokenizer有新tokens，需要调整embedding层大小
        if tokenizer is not None:
            self.backbone.resize_token_embeddings(len(tokenizer))
            print(f"调整模型embedding层大小为: {len(tokenizer)}")
        
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS] token的输出进行分类
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def train_model_npu():
    # 设置分布式训练
    is_distributed, rank, world_size, device_id = setup_distributed_npu()
    
    # 加载配置
    config = TrainingConfigNPU()
    
    # 设置设备
    if NPU_AVAILABLE:
        device = torch.device(f'npu:{device_id}')
        if is_main_process(rank):
            if is_distributed:
                print(f"使用昇腾NPU分布式训练，总进程数: {world_size}, 当前进程: {rank}, NPU: {device_id}")
            else:
                print(f"使用单NPU训练，NPU: {device}")
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        if is_main_process(rank):
            print(f"昇腾NPU不可用，使用GPU训练")
    else:
        device = torch.device('cpu')
        if is_main_process(rank):
            print("使用CPU训练")
    
    # 创建数据加载器
    if is_main_process(rank):
        print("创建数据加载器...")
    train_loader, val_loader, tokenizer = create_data_loaders(
        config, 
        is_distributed=is_distributed, 
        rank=rank, 
        world_size=world_size
    )
    
    # 创建模型
    if is_main_process(rank):
        print("加载模型...")
    model = TrajectoryClassifier(config.model_name, config.num_labels, tokenizer)
    
    # NPU模型适配
    if NPU_AVAILABLE:
        model = model.to(device)
        # 昇腾NPU的混合精度训练
        if hasattr(config, 'fp16') and config.fp16:
            model = model.half()  # 转换为半精度
    else:
        model = model.to(device)
    
    # 包装为分布式模型
    if is_distributed:
        if NPU_AVAILABLE:
            # 昇腾NPU使用的DDP配置
            model = DDP(model, device_ids=[device_id], broadcast_buffers=False)
        else:
            model = DDP(model, device_ids=[device_id], output_device=device_id, find_unused_parameters=True)
    
    # 设置优化器和学习率调度器
    # 昇腾NPU推荐使用特定的优化器设置
    if NPU_AVAILABLE:
        optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建输出目录（只在主进程创建）
    if is_main_process(rank):
        os.makedirs(config.output_dir, exist_ok=True)
        print("开始训练...")
    
    best_val_acc = 0
    
    for epoch in range(config.num_epochs):
        if is_main_process(rank):
            print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
        
        # 分布式训练需要设置epoch以确保数据打乱
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # 使用tqdm进度条（只在主进程显示）
        train_iter = tqdm(train_loader, desc="Training") if is_main_process(rank) else train_loader
        
        for batch_idx, batch in enumerate(train_iter):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # NPU混合精度训练的数据类型处理
            if NPU_AVAILABLE and hasattr(config, 'fp16') and config.fp16:
                input_ids = input_ids.half()
                attention_mask = attention_mask.half()
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if is_main_process(rank) and (batch_idx + 1) % config.logging_steps == 0:
                print(f"Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
        
        # 聚合所有进程的训练指标
        if is_distributed:
            # 创建张量用于all_reduce操作
            if NPU_AVAILABLE:
                train_loss_tensor = torch.tensor(train_loss, device=device, dtype=torch.float32)
                train_correct_tensor = torch.tensor(train_correct, device=device, dtype=torch.int64)
                train_total_tensor = torch.tensor(train_total, device=device, dtype=torch.int64)
            else:
                train_loss_tensor = torch.tensor(train_loss, device=device)
                train_correct_tensor = torch.tensor(train_correct, device=device)
                train_total_tensor = torch.tensor(train_total, device=device)
            
            # 聚合所有进程的结果
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
            
            train_loss = train_loss_tensor.item()
            train_correct = train_correct_tensor.item()
            train_total = train_total_tensor.item()
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader) / (world_size if is_distributed else 1)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        # 验证时不需要进度条在所有进程显示
        val_iter = tqdm(val_loader, desc="Validation") if is_main_process(rank) else val_loader
        
        with torch.no_grad():
            for batch in val_iter:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # NPU混合精度训练的数据类型处理
                if NPU_AVAILABLE and hasattr(config, 'fp16') and config.fp16:
                    input_ids = input_ids.half()
                    attention_mask = attention_mask.half()
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 聚合验证指标
        if is_distributed:
            if NPU_AVAILABLE:
                val_loss_tensor = torch.tensor(val_loss, device=device, dtype=torch.float32)
                val_correct_tensor = torch.tensor(val_correct, device=device, dtype=torch.int64)
                val_total_tensor = torch.tensor(val_total, device=device, dtype=torch.int64)
            else:
                val_loss_tensor = torch.tensor(val_loss, device=device)
                val_correct_tensor = torch.tensor(val_correct, device=device)
                val_total_tensor = torch.tensor(val_total, device=device)
            
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
            
            val_loss = val_loss_tensor.item()
            val_correct = val_correct_tensor.item()
            val_total = val_total_tensor.item()
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader) / (world_size if is_distributed else 1)
        
        if is_main_process(rank):
            print(f"训练损失: {avg_train_loss:.4f}, 训练精度: {train_acc:.4f}")
            print(f"验证损失: {avg_val_loss:.4f}, 验证精度: {val_acc:.4f}")
        
        # 保存最佳模型（只在主进程保存）
        if is_main_process(rank) and val_acc > best_val_acc:
            best_val_acc = val_acc
            # 如果是分布式训练，保存model.module的状态而不是DDP包装的模型
            model_state_dict = model.module.state_dict() if is_distributed else model.state_dict()
            
            # 昇腾NPU模型保存时需要转换回CPU
            if NPU_AVAILABLE:
                model_state_dict = {k: v.cpu() for k, v in model_state_dict.items()}
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'device_type': 'npu' if NPU_AVAILABLE else 'gpu'
            }, os.path.join(config.output_dir, 'best_model.pt'))
            print(f"保存最佳模型，验证精度: {val_acc:.4f}")
        
        # 同步所有进程
        if is_distributed:
            dist.barrier()
    
    if is_main_process(rank):
        print(f"\n训练完成！最佳验证精度: {best_val_acc:.4f}")
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(all_labels, all_predictions))
    
    # 清理分布式训练环境
    cleanup_distributed()
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = train_model_npu()