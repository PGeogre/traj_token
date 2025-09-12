# 轨迹NPU微调脚本
import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
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
import argparse
from sklearn.metrics import classification_report, accuracy_score

from pretrain_config_npu import FineTuneConfigNPU, ConfigManagerNPU
from data_loader import create_data_loaders
from pretrain_models import TrajectoryClassificationModel

def setup_distributed_npu():
    """设置NPU分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("未检测到分布式环境变量，将使用单NPU训练")
        return False, 0, 1, 0
    
    # 设置当前NPU设备
    torch.npu.set_device(local_rank)
    
    # 初始化分布式进程组
    dist.init_process_group(
        backend='hccl',  # 华为NPU使用HCCL后端
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return True, rank, world_size, local_rank

def cleanup_distributed_npu():
    """清理NPU分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    """判断是否为主进程"""
    return rank == 0

def validate_model_npu(model, val_loader, device, is_distributed, rank, world_size):
    """NPU模型验证"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        val_iter = tqdm(val_loader, desc="验证中") if is_main_process(rank) else val_loader
        
        for batch in val_iter:
            # 将数据转移到NPU
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            outputs = model(input_ids, attention_mask, labels)
            
            loss = outputs.loss
            logits = outputs.logits
            
            # 收集预测和标签
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            total_samples += 1
    
    # 聚合所有进程的验证指标
    if is_distributed:
        # 收集所有进程的预测和标签
        predictions_tensor = torch.tensor(all_predictions, device=device)
        labels_tensor = torch.tensor(all_labels, device=device)
        
        # 收集到所有进程
        predictions_list = [torch.zeros_like(predictions_tensor) for _ in range(world_size)]
        labels_list = [torch.zeros_like(labels_tensor) for _ in range(world_size)]
        
        dist.all_gather(predictions_list, predictions_tensor)
        dist.all_gather(labels_list, labels_tensor)
        
        if is_main_process(rank):
            all_predictions = torch.cat(predictions_list).cpu().numpy()
            all_labels = torch.cat(labels_list).cpu().numpy()
        
        # 聚合损失
        loss_tensor = torch.tensor(total_loss, device=device)
        samples_tensor = torch.tensor(total_samples, device=device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = loss_tensor.item()
        total_samples = samples_tensor.item()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    if is_main_process(rank):
        accuracy = accuracy_score(all_labels, all_predictions)
        return avg_loss, accuracy, all_predictions, all_labels
    else:
        return avg_loss, 0, [], []

def save_training_log(log_data, log_file):
    """保存训练日志"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_data) + '\n')

def finetune_model_npu(pretrained_model_path=None):
    """主要的NPU微调函数"""
    
    # 设置NPU分布式训练
    is_distributed, rank, world_size, local_rank = setup_distributed_npu()
    
    # 加载配置
    config = FineTuneConfigNPU()
    
    # 如果提供了预训练模型路径，使用该路径
    if pretrained_model_path:
        config.pretrained_model_path = pretrained_model_path
    
    if is_main_process(rank):
        ConfigManagerNPU.setup_npu_environment()
        ConfigManagerNPU.print_config(config, "微调阶段")
    
    # 设置设备
    if is_distributed:
        device = torch.device(f'npu:{local_rank}')
        if is_main_process(rank):
            print(f"使用分布式NPU微调，总进程数: {world_size}, 当前进程: {rank}, NPU: {local_rank}")
    else:
        device = torch.device('npu:0' if torch_npu.npu.is_available() else 'cpu')
        if torch_npu.npu.is_available():
            print(f"使用单NPU微调，设备: {device}")
        else:
            print("NPU不可用，使用CPU训练")
    
    # 检查预训练模型是否存在
    if not os.path.exists(config.pretrained_model_path):
        raise FileNotFoundError(f"预训练模型不存在: {config.pretrained_model_path}")
    
    # 创建输出目录
    if is_main_process(rank):
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"输出目录: {config.output_dir}")
        print(f"预训练模型路径: {config.pretrained_model_path}")
    
    # 创建数据加载器
    if is_main_process(rank):
        print("创建数据加载器...")
    
    # 使用原有的数据加载器，但需要适配NPU
    config.model_name = config.pretrained_model_path  # 使用预训练模型路径
    train_loader, val_loader, tokenizer = create_data_loaders(
        config, is_distributed, rank, world_size
    )
    
    # 创建分类模型
    if is_main_process(rank):
        print("创建NPU微调分类模型...")
    
    model = TrajectoryClassificationModel(
        config.pretrained_model_path, 
        config.num_labels,
        config.dropout_rate
    )
    model.to(device)
    
    if is_main_process(rank):
        from pretrain_models import TrajectoryModelUtils
        TrajectoryModelUtils.count_parameters(model)
        TrajectoryModelUtils.get_model_size_mb(model)
    
    # 包装为分布式模型
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, 
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
    
    # NPU混合精度训练
    scaler = torch.npu.amp.GradScaler() if config.use_amp else None
    
    # 训练日志
    if is_main_process(rank):
        log_file = os.path.join(config.output_dir, 'training_log.jsonl')
        print("开始NPU微调...")
        print(f"总训练步数: {total_steps}")
        print(f"使用混合精度: {config.use_amp}")
    
    best_val_accuracy = 0
    patience_counter = 0
    global_step = 0
    
    for epoch in range(config.num_epochs):
        if is_main_process(rank):
            print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
        
        # 分布式训练需要设置epoch
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # 冻结backbone（如果配置了）
        if epoch < config.freeze_backbone_epochs:
            if is_main_process(rank):
                print(f"冻结backbone，epoch {epoch + 1}/{config.freeze_backbone_epochs}")
            # 冻结除分类头外的所有参数
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        else:
            # 解冻所有参数
            for param in model.parameters():
                param.requires_grad = True
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_samples = 0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} NPU微调") if is_main_process(rank) else train_loader
        
        for batch_idx, batch in enumerate(train_iter):
            # 将数据转移到NPU
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # 前向传播
            if config.use_amp:
                with torch.npu.amp.autocast():
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs.loss / config.gradient_accumulation_steps
            else:
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs.loss / config.gradient_accumulation_steps
            
            # 反向传播
            if config.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            train_loss += loss.item() * config.gradient_accumulation_steps
            train_samples += 1
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # 梯度裁剪和优化器步骤
                if config.use_amp:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 日志记录
                if is_main_process(rank) and global_step % config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    avg_train_loss = train_loss / train_samples
                    print(f"Step {global_step}, Loss: {avg_train_loss:.4f}, LR: {current_lr:.2e}")
                
                # 验证
                if global_step % config.eval_steps == 0:
                    val_loss, val_accuracy, predictions, labels_true = validate_model_npu(
                        model, val_loader, device, is_distributed, rank, world_size
                    )
                    
                    if is_main_process(rank):
                        print(f"验证 - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
                        
                        # 保存训练日志
                        log_data = {
                            'epoch': epoch + 1,
                            'step': global_step,
                            'train_loss': train_loss / train_samples,
                            'val_loss': val_loss,
                            'val_accuracy': val_accuracy,
                            'learning_rate': scheduler.get_last_lr()[0],
                            'timestamp': datetime.now().isoformat()
                        }
                        save_training_log(log_data, log_file)
                        
                        # 打印详细分类报告
                        if len(predictions) > 0 and len(labels_true) > 0:
                            report = classification_report(
                                labels_true, predictions, 
                                target_names=[f"Class_{i}" for i in range(config.num_labels)],
                                digits=4,
                                output_dict=True,
                                zero_division=0
                            )
                            print(f"宏平均F1: {report['macro avg']['f1-score']:.4f}")
                            print(f"加权平均F1: {report['weighted avg']['f1-score']:.4f}")
                    
                    # 早停检查（基于准确率）
                    if val_accuracy > best_val_accuracy + config.early_stopping_threshold:
                        best_val_accuracy = val_accuracy
                        patience_counter = 0
                        
                        # 保存最佳模型
                        if is_main_process(rank):
                            save_path = os.path.join(config.output_dir, 'best_model')
                            if is_distributed:
                                model.module.save_pretrained(save_path)
                            else:
                                model.save_pretrained(save_path)
                            
                            tokenizer.save_pretrained(save_path)
                            
                            # 保存验证报告
                            if len(predictions) > 0 and len(labels_true) > 0:
                                report_path = os.path.join(save_path, 'validation_report.json')
                                with open(report_path, 'w') as f:
                                    json.dump(report, f, indent=2)
                            
                            print(f"保存最佳NPU模型，验证准确率: {val_accuracy:.4f}")
                    else:
                        patience_counter += 1
                    
                    # 早停
                    if patience_counter >= config.early_stopping_patience:
                        if is_main_process(rank):
                            print(f"早停：验证准确率在 {config.early_stopping_patience} 步内未改善")
                        break
                    
                    model.train()  # 恢复训练模式
                
                # 定期保存检查点
                if is_main_process(rank) and global_step % config.save_steps == 0:
                    save_path = os.path.join(config.output_dir, f'checkpoint-{global_step}')
                    if is_distributed:
                        model.module.save_pretrained(save_path)
                    else:
                        model.save_pretrained(save_path)
                    
                    tokenizer.save_pretrained(save_path)
                    print(f"保存NPU检查点: {save_path}")
        
        # 如果早停，跳出epoch循环
        if patience_counter >= config.early_stopping_patience:
            break
        
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
        print(f"NPU微调完成！最终模型保存至: {final_save_path}")
        
        # 最终验证
        print("进行最终验证...")
        final_val_loss, final_val_accuracy, final_predictions, final_labels = validate_model_npu(
            model, val_loader, device, is_distributed, rank, world_size
        )
        
        print(f"最终验证结果 - Loss: {final_val_loss:.4f}, Accuracy: {final_val_accuracy:.4f}")
        
        if len(final_predictions) > 0 and len(final_labels) > 0:
            final_report = classification_report(
                final_labels, final_predictions,
                target_names=[f"Class_{i}" for i in range(config.num_labels)],
                digits=4,
                output_dict=True,
                zero_division=0
            )
            
            print("最终分类报告:")
            print(f"  整体准确率: {final_report['accuracy']:.4f}")
            print(f"  宏平均F1: {final_report['macro avg']['f1-score']:.4f}")
            print(f"  加权平均F1: {final_report['weighted avg']['f1-score']:.4f}")
            
            # 保存最终报告
            final_report_path = os.path.join(final_save_path, 'final_validation_report.json')
            with open(final_report_path, 'w') as f:
                json.dump(final_report, f, indent=2)
    
    # 清理分布式训练环境
    cleanup_distributed_npu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NPU微调训练脚本')
    parser.add_argument('--pretrained_model_path', type=str, default=None,
                        help='预训练模型路径')
    
    args = parser.parse_args()
    finetune_model_npu(args.pretrained_model_path)