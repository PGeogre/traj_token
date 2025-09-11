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

from train_config import TrainingConfig
from data_loader import create_data_loaders

def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_id = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
        gpu_id = rank % torch.cuda.device_count()
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

def train_model():
    # 设置分布式训练
    is_distributed, rank, world_size, gpu_id = setup_distributed()
    
    # 加载配置
    config = TrainingConfig()
    
    # 设置设备
    if is_distributed:
        device = torch.device(f'cuda:{gpu_id}')
        if is_main_process(rank):
            print(f"使用分布式训练，总进程数: {world_size}, 当前进程: {rank}, GPU: {gpu_id}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"使用单卡训练，GPU: {device}")
        else:
            print("使用CPU训练")
    
    # 创建数据加载器（需要修改data_loader.py以支持分布式）
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
    model.to(device)
    
    # 包装为分布式模型
    if is_distributed:
        model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True)
    
    # 设置优化器和学习率调度器
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
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
    model, tokenizer = train_model()