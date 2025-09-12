# 专门的微调训练脚本
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pretrain_config import FineTuneConfig, ConfigManager
from data_loader import create_data_loaders
from pretrain_models import TrajectoryFineTuneModel, TrajectoryModelUtils

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

def create_finetune_data_loaders(config, is_distributed=False, rank=0, world_size=1):
    """创建微调数据加载器 - 复用原来的data_loader逻辑但调整配置"""
    
    # 临时修改配置以匹配原始data_loader的接口
    temp_config = type('TempConfig', (), {})()
    temp_config.model_name = "/home/maomao/pretrained_model/qwen2.5-0.5b"  # tokenizer路径
    temp_config.train_data_file = config.train_data_file
    temp_config.max_length = config.max_length
    temp_config.batch_size = config.batch_size
    temp_config.dataloader_num_workers = config.dataloader_num_workers
    temp_config.num_labels = config.num_labels
    
    return create_data_loaders(temp_config, is_distributed, rank, world_size)

def validate_model(model, val_loader, device, is_distributed, rank, world_size):
    """验证模型性能"""
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_labels = []
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        val_iter = tqdm(val_loader, desc="验证中") if is_main_process(rank) else val_loader
        
        for batch in val_iter:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            predictions = torch.argmax(logits, dim=-1)
            
            total_loss += loss.item()
            total_samples += 1
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 聚合所有进程的验证指标
    if is_distributed:
        # 收集所有预测和标签
        predictions_tensor = torch.tensor(all_predictions, device=device)
        labels_tensor = torch.tensor(all_labels, device=device)
        loss_tensor = torch.tensor(total_loss, device=device)
        samples_tensor = torch.tensor(total_samples, device=device)
        
        # 创建收集列表
        gathered_predictions = [torch.zeros_like(predictions_tensor) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]
        
        # 收集数据
        dist.all_gather(gathered_predictions, predictions_tensor)
        dist.all_gather(gathered_labels, labels_tensor)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        
        # 只在主进程处理结果
        if is_main_process(rank):
            all_predictions = torch.cat(gathered_predictions).cpu().numpy()
            all_labels = torch.cat(gathered_labels).cpu().numpy()
        
        total_loss = loss_tensor.item()
        total_samples = samples_tensor.item()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    # 计算准确率和其他指标
    if is_main_process(rank):
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # 生成详细报告
        class_names = [
            'Bulk_Carrier', 'Cargo_Ship', 'Container_Ship', 'Barge',
            'Fishing_Vessel', 'Other', 'Oil_Tanker', 'Passenger_Ship',
            'Sand_Carrier', 'Fishery_Research_Vessel', 'Supply_Ship',
            'Storage_Tanker', 'Submarine', 'Transport_Ship'
        ]
        
        report = classification_report(
            all_labels, all_predictions, 
            target_names=class_names[:max(all_labels)+1],
            output_dict=True,
            zero_division=0
        )
        
        return avg_loss, accuracy, report, all_predictions, all_labels
    else:
        return avg_loss, 0, {}, [], []

def save_training_log(log_data, log_file):
    """保存训练日志"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

def finetune_model(pretrained_model_path=None):
    """主要的微调函数"""
    
    # 设置分布式训练
    is_distributed, rank, world_size, gpu_id = setup_distributed()
    
    # 加载配置
    config = FineTuneConfig()
    
    if pretrained_model_path:
        config.pretrained_model_path = pretrained_model_path
    
    if is_main_process(rank):
        ConfigManager.print_config(config, "微调阶段")
    
    # 设置设备
    if is_distributed:
        device = torch.device(f'cuda:{gpu_id}')
        if is_main_process(rank):
            print(f"使用分布式微调训练，总进程数: {world_size}, 当前进程: {rank}, GPU: {gpu_id}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"使用单卡微调训练，GPU: {device}")
        else:
            print("使用CPU微调训练")
    
    # 创建输出目录
    if is_main_process(rank):
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"输出目录: {config.output_dir}")
    
    # 创建数据加载器
    if is_main_process(rank):
        print("创建数据加载器...")
    
    train_loader, val_loader, tokenizer = create_finetune_data_loaders(
        config, is_distributed, rank, world_size
    )
    
    # 创建微调模型
    if is_main_process(rank):
        print("创建微调模型...")
        print(f"预训练模型路径: {config.pretrained_model_path}")
    
    model = TrajectoryFineTuneModel(
        config.pretrained_model_path,
        config.num_labels,
        tokenizer,
        config.dropout_rate
    )
    model.to(device)
    
    # 打印模型信息
    if is_main_process(rank):
        TrajectoryModelUtils.count_parameters(model)
        TrajectoryModelUtils.get_model_size_mb(model)
    
    # 可选择性冻结backbone
    if config.freeze_backbone_epochs > 0:
        model.freeze_backbone(freeze=True)
        if is_main_process(rank):
            print(f"前 {config.freeze_backbone_epochs} 轮将冻结backbone")
    
    # 包装为分布式模型
    if is_distributed:
        model = DDP(model, device_ids=[gpu_id], output_device=gpu_id,
                   find_unused_parameters=config.find_unused_parameters)
    
    # 设置优化器 - 对不同部分使用不同学习率
    if hasattr(model, 'module'):
        model_for_optim = model.module
    else:
        model_for_optim = model
    
    # 分层学习率：backbone用较小学习率，分类头用标准学习率
    optimizer_grouped_parameters = [
        {
            'params': model_for_optim.backbone.parameters(),
            'lr': config.learning_rate * 0.1,  # backbone用1/10的学习率
            'weight_decay': config.weight_decay
        },
        {
            'params': [model_for_optim.classifier.weight, model_for_optim.classifier.bias],
            'lr': config.learning_rate,
            'weight_decay': 0.0  # 分类层不用权重衰减
        }
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
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
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练日志
    if is_main_process(rank):
        log_file = os.path.join(config.output_dir, 'finetune_log.jsonl')
        print("开始微调...")
        print(f"总训练步数: {total_steps}")
    
    best_val_acc = 0
    patience_counter = 0
    global_step = 0
    
    for epoch in range(config.num_epochs):
        if is_main_process(rank):
            print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
        
        # 检查是否需要解冻backbone
        if epoch == config.freeze_backbone_epochs and config.freeze_backbone_epochs > 0:
            if hasattr(model, 'module'):
                model.module.freeze_backbone(freeze=False)
            else:
                model.freeze_backbone(freeze=False)
            if is_main_process(rank):
                print("解冻backbone，开始端到端微调")
        
        # 分布式训练需要设置epoch
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} 微调") if is_main_process(rank) else train_loader
        
        for batch_idx, batch in enumerate(train_iter):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss'] / config.gradient_accumulation_steps
            
            loss.backward()
            
            train_loss += loss.item() * config.gradient_accumulation_steps
            
            # 计算准确率
            logits = outputs['logits']
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 日志记录
                if is_main_process(rank) and global_step % config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()
                    print(f"Step {global_step}, Loss: {loss.item():.4f}, "
                          f"LR_backbone: {current_lr[0]:.2e}, LR_classifier: {current_lr[1]:.2e}")
                
                # 验证
                if global_step % config.eval_steps == 0:
                    val_loss, val_accuracy, val_report, val_predictions, val_labels = validate_model(
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
                            'learning_rates': scheduler.get_last_lr(),
                            'timestamp': datetime.now().isoformat()
                        }
                        if val_report:
                            log_data['val_report'] = val_report
                        
                        save_training_log(log_data, log_file)
                    
                    # 保存最佳模型
                    if val_accuracy > best_val_acc:
                        best_val_acc = val_accuracy
                        patience_counter = 0
                        
                        if is_main_process(rank):
                            save_path = os.path.join(config.output_dir, 'best_model')
                            os.makedirs(save_path, exist_ok=True)
                            
                            # 保存模型权重
                            model_state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                            torch.save({
                                'model_state_dict': model_state_dict,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch,
                                'val_accuracy': val_accuracy,
                                'val_loss': val_loss,
                                'config': config.__dict__
                            }, os.path.join(save_path, 'pytorch_model.bin'))
                            
                            # 保存tokenizer
                            tokenizer.save_pretrained(save_path)
                            
                            # 保存详细结果
                            if val_report:
                                with open(os.path.join(save_path, 'validation_report.json'), 'w') as f:
                                    json.dump(val_report, f, indent=2, ensure_ascii=False)
                            
                            print(f"保存最佳模型，验证精度: {val_accuracy:.4f}")
                    else:
                        patience_counter += 1
                    
                    # 早停
                    if patience_counter >= config.early_stopping_patience:
                        if is_main_process(rank):
                            print(f"早停：验证精度在 {config.early_stopping_patience} 步内未改善")
                        return
                    
                    model.train()  # 恢复训练模式
        
        # 聚合训练指标
        if is_distributed:
            train_loss_tensor = torch.tensor(train_loss, device=device)
            train_correct_tensor = torch.tensor(train_correct, device=device)
            train_total_tensor = torch.tensor(train_total, device=device)
            
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
            
            train_loss = train_loss_tensor.item()
            train_correct = train_correct_tensor.item()
            train_total = train_total_tensor.item()
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader) / (world_size if is_distributed else 1)
        
        if is_main_process(rank):
            print(f"Epoch {epoch + 1} - 训练损失: {avg_train_loss:.4f}, 训练精度: {train_acc:.4f}")
        
        # 同步所有进程
        if is_distributed:
            dist.barrier()
    
    # 训练结束
    if is_main_process(rank):
        print(f"微调完成！最佳验证精度: {best_val_acc:.4f}")
        
        # 保存最终模型
        final_save_path = os.path.join(config.output_dir, 'final_model')
        os.makedirs(final_save_path, exist_ok=True)
        
        model_state_dict = model.module.state_dict() if is_distributed else model.state_dict()
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_accuracy': best_val_acc,
            'config': config.__dict__
        }, os.path.join(final_save_path, 'pytorch_model.bin'))
        
        tokenizer.save_pretrained(final_save_path)
        print(f"最终模型保存至: {final_save_path}")
    
    # 清理分布式训练环境
    cleanup_distributed()
    
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='轨迹模型微调')
    parser.add_argument('--pretrained_model_path', type=str,
                       default="../model_out/trajectory_pretrain_model/best_model",
                       help='预训练模型路径')
    parser.add_argument('--local-rank', type=int, default=-1,
                       help='分布式训练的local rank参数')
    
    args = parser.parse_args()
    
    model, tokenizer = finetune_model(args.pretrained_model_path)