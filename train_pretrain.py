# 轨迹预训练脚本
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import numpy as np
from datetime import datetime

from pretrain_config import PretrainConfig, ConfigManager
from pretrain_data_processor import create_pretrain_dataset, PretrainDataset
from pretrain_models import TrajectoryPretrainModel

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

def create_pretrain_data_loaders(config, tokenizer, is_distributed=False, rank=0, world_size=1):
    """创建预训练数据加载器"""
    
    # 检查是否需要生成预训练数据
    if not os.path.exists(config.pretrain_data_file):
        if is_main_process(rank):
            print("预训练数据文件不存在，正在生成...")
            from pretrain_data_processor import PretrainDataProcessor
            processor = PretrainDataProcessor(tokenizer)
            processor.create_pretrain_data(
                config.train_data_file,
                config.pretrain_data_file,
                mask_prob=config.mask_prob
            )
        
        # 在分布式训练中，等待主进程完成数据生成
        if is_distributed:
            dist.barrier()
    
    # 创建数据集
    if is_main_process(rank):
        print(f"加载预训练数据: {config.pretrain_data_file}")
    
    full_dataset = PretrainDataset(
        config.pretrain_data_file,
        tokenizer,
        config.max_length
    )
    
    # 分割训练和验证集
    val_size = int(config.eval_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子确保一致性
    )
    
    # 创建数据加载器
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.dataloader_num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=config.dataloader_num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.dataloader_num_workers
        )
    
    if is_main_process(rank):
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        if is_distributed:
            print(f"分布式训练: 每个GPU处理 ~{len(train_dataset)//world_size} 个训练样本")
    
    return train_loader, val_loader

def validate_model(model, val_loader, device, is_distributed, rank, world_size):
    """验证模型性能"""
    model.eval()
    
    total_loss = 0
    total_accuracy = 0
    total_samples = 0
    
    with torch.no_grad():
        val_iter = tqdm(val_loader, desc="验证中") if is_main_process(rank) else val_loader
        
        for batch in val_iter:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            
            loss = outputs['loss']
            accuracy = outputs['accuracy']
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_samples += 1
    
    # 聚合所有进程的验证指标
    if is_distributed:
        loss_tensor = torch.tensor(total_loss, device=device)
        accuracy_tensor = torch.tensor(total_accuracy, device=device)
        samples_tensor = torch.tensor(total_samples, device=device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = loss_tensor.item()
        total_accuracy = accuracy_tensor.item()
        total_samples = samples_tensor.item()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
    
    return avg_loss, avg_accuracy

def save_training_log(log_data, log_file):
    """保存训练日志"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_data) + '\n')

def pretrain_model():
    """主要的预训练函数"""
    
    # 设置分布式训练
    is_distributed, rank, world_size, gpu_id = setup_distributed()
    
    # 加载配置
    config = PretrainConfig()
    
    if is_main_process(rank):
        ConfigManager.print_config(config, "预训练阶段")
    
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
    
    # 创建输出目录
    if is_main_process(rank):
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"输出目录: {config.output_dir}")
    
    # 生成预训练数据并创建tokenizer
    if is_main_process(rank):
        print("准备预训练数据和tokenizer...")
    
    pretrain_data_file, tokenizer = create_pretrain_dataset(config)
    
    # 在分布式训练中同步tokenizer
    if is_distributed:
        dist.barrier()  # 等待所有进程
    
    # 创建数据加载器
    train_loader, val_loader = create_pretrain_data_loaders(
        config, tokenizer, is_distributed, rank, world_size
    )
    
    # 创建模型
    if is_main_process(rank):
        print("创建预训练模型...")
    
    model = TrajectoryPretrainModel(config.model_name, tokenizer)
    model.to(device)
    
    if is_main_process(rank):
        from pretrain_models import TrajectoryModelUtils
        TrajectoryModelUtils.count_parameters(model)
        TrajectoryModelUtils.get_model_size_mb(model)
    
    # 包装为分布式模型
    if is_distributed:
        model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, 
                   find_unused_parameters=config.find_unused_parameters)
    
    # 设置优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon
    )
    
    # 学习率调度器
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练日志
    if is_main_process(rank):
        log_file = os.path.join(config.output_dir, 'training_log.jsonl')
        print("开始预训练...")
        print(f"总训练步数: {total_steps}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    
    for epoch in range(config.num_epochs):
        if is_main_process(rank):
            print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
        
        # 分布式训练需要设置epoch
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_accuracy = 0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} 训练") if is_main_process(rank) else train_loader
        
        for batch_idx, batch in enumerate(train_iter):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss'] / config.gradient_accumulation_steps
            accuracy = outputs['accuracy']
            
            loss.backward()
            
            train_loss += loss.item() * config.gradient_accumulation_steps
            train_accuracy += accuracy.item()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 日志记录
                if is_main_process(rank) and global_step % config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
                
                # 验证
                if global_step % config.eval_steps == 0:
                    val_loss, val_accuracy = validate_model(
                        model, val_loader, device, is_distributed, rank, world_size
                    )
                    
                    if is_main_process(rank):
                        print(f"验证 - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
                        
                        # 保存训练日志
                        log_data = {
                            'epoch': epoch + 1,
                            'step': global_step,
                            'train_loss': loss.item(),
                            'val_loss': val_loss,
                            'val_accuracy': val_accuracy,
                            'learning_rate': scheduler.get_last_lr()[0],
                            'timestamp': datetime.now().isoformat()
                        }
                        save_training_log(log_data, log_file)
                    
                    # 早停检查
                    if val_loss < best_val_loss - config.early_stopping_threshold:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # 保存最佳模型
                        if is_main_process(rank):
                            save_path = os.path.join(config.output_dir, 'best_model')
                            if is_distributed:
                                model.module.save_pretrained(save_path)
                            else:
                                model.save_pretrained(save_path)
                            
                            tokenizer.save_pretrained(save_path)
                            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
                    else:
                        patience_counter += 1
                    
                    # 早停
                    if patience_counter >= config.early_stopping_patience:
                        if is_main_process(rank):
                            print(f"早停：验证损失在 {config.early_stopping_patience} 步内未改善")
                        return
                    
                    model.train()  # 恢复训练模式
                
                # 定期保存检查点
                if is_main_process(rank) and global_step % config.save_steps == 0:
                    save_path = os.path.join(config.output_dir, f'checkpoint-{global_step}')
                    if is_distributed:
                        model.module.save_pretrained(save_path)
                    else:
                        model.save_pretrained(save_path)
                    
                    tokenizer.save_pretrained(save_path)
                    print(f"保存检查点: {save_path}")
        
        # 同步所有进程
        if is_distributed:
            dist.barrier()
    
    # 训练结束，保存最终模型
    if is_main_process(rank):
        final_save_path = os.path.join(config.output_dir, 'final_model')
        if is_distributed:
            model.module.save_pretrained(final_save_path)
        else:
            model.save_pretrained(final_save_path)
        
        tokenizer.save_pretrained(final_save_path)
        print(f"预训练完成！最终模型保存至: {final_save_path}")
    
    # 清理分布式训练环境
    cleanup_distributed()

if __name__ == "__main__":
    pretrain_model()