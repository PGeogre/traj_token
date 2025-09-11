# 预训练数据处理模块
import json
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np

class PretrainDataProcessor:
    """预训练数据处理类，用于创建MLM任务的掩码数据"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.mask_token
        self.mask_token_id = tokenizer.mask_token_id
        
        # 定义不同类型的tokens
        self.h3_prefixes = ['H3_L', 'H3_GRID', 'H3_BASE', 'H3_SUB', 'H3_PRECISE', 'GRID_']
        self.vessel_class_tokens = [
            'Bulk_Carrier', 'Cargo_Ship', 'Container_Ship', 'Barge',
            'Fishing_Vessel', 'Other', 'Oil_Tanker', 'Passenger_Ship',
            'Sand_Carrier', 'Fishery_Research_Vessel', 'Supply_Ship',
            'Storage_Tanker', 'Submarine', 'Transport_Ship'
        ]
        self.motion_prefixes = ['SPD_', 'COG_']
        
        # 掩码策略权重
        self.h3_mask_prob = 0.4     # H3地理位置掩码概率
        self.class_mask_prob = 0.3  # 船舶类别掩码概率  
        self.motion_mask_prob = 0.2 # 速度/航向掩码概率
        self.random_mask_prob = 0.1 # 随机掩码其他tokens的概率
    
    def should_mask_token(self, token):
        """判断token是否应该被掩码"""
        # H3位置相关tokens
        if any(token.startswith(prefix) for prefix in self.h3_prefixes):
            return random.random() < self.h3_mask_prob
        
        # 船舶类别tokens
        if token in self.vessel_class_tokens:
            return random.random() < self.class_mask_prob
        
        # 速度和航向tokens  
        if any(token.startswith(prefix) for prefix in self.motion_prefixes):
            return random.random() < self.motion_mask_prob
        
        # 随机掩码其他tokens（但保留时间和结构化tokens）
        if not token.startswith(('YEAR_', 'MONTH_', 'DAY_', 'HOUR_', 'MINUTE_', 'SECOND_', 'POINT_END')):
            return random.random() < self.random_mask_prob
        
        return False
    
    def mask_tokens(self, tokens, mask_prob=0.15):
        """
        对token序列进行掩码处理
        
        Args:
            tokens: token列表
            mask_prob: 总体掩码概率
            
        Returns:
            masked_tokens: 掩码后的token列表
            labels: 用于计算MLM loss的标签（-100表示不计算loss）
        """
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)
        
        for i, token in enumerate(tokens):
            # 应用掩码策略
            if self.should_mask_token(token) and random.random() < mask_prob:
                # 80%概率用[MASK]替换，10%概率用随机token，10%概率保持原样
                rand = random.random()
                
                if rand < 0.8:
                    # 用[MASK]替换
                    masked_tokens[i] = self.mask_token
                elif rand < 0.9:
                    # 用随机token替换（从词汇表中随机选择）
                    vocab_size = len(self.tokenizer)
                    random_token_id = random.randint(0, vocab_size - 1)
                    masked_tokens[i] = self.tokenizer.convert_ids_to_tokens(random_token_id)
                # 10%概率保持原token不变
                
                # 设置标签为原始token的ID（用于计算loss）
                try:
                    labels[i] = self.tokenizer.convert_tokens_to_ids(token)
                except:
                    # 如果token不在词汇表中，跳过
                    labels[i] = -100
        
        return masked_tokens, labels
    
    def create_pretrain_data(self, jsonl_file, output_file, mask_prob=0.15):
        """
        从原始数据创建预训练数据
        
        Args:
            jsonl_file: 原始数据文件路径
            output_file: 输出的预训练数据文件路径
            mask_prob: 掩码概率
        """
        print(f"开始处理预训练数据：{jsonl_file}")
        print(f"掩码概率：{mask_prob}")
        print(f"掩码策略：H3={self.h3_mask_prob}, 类别={self.class_mask_prob}, 运动={self.motion_mask_prob}")
        
        total_samples = 0
        masked_tokens_count = 0
        
        with open(jsonl_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile):
                if line_num % 1000 == 0:
                    print(f"处理第 {line_num} 行...")
                
                try:
                    item = json.loads(line.strip())
                    tokens = item['text'].split()
                    
                    # 对tokens进行掩码
                    masked_tokens, labels = self.mask_tokens(tokens, mask_prob)
                    
                    # 统计掩码token数量
                    masked_count = sum(1 for label in labels if label != -100)
                    masked_tokens_count += masked_count
                    
                    # 创建预训练样本
                    pretrain_sample = {
                        'input_text': ' '.join(masked_tokens),
                        'labels': labels,
                        'original_text': item['text'],
                        'original_label': item.get('label', -1),  # 保留原始分类标签
                        'source_file': item.get('source_file', ''),
                        'masked_tokens_count': masked_count,
                        'total_tokens_count': len(tokens)
                    }
                    
                    outfile.write(json.dumps(pretrain_sample) + '\n')
                    total_samples += 1
                    
                except Exception as e:
                    print(f"处理第 {line_num} 行时出错: {e}")
        
        print(f"预训练数据创建完成！")
        print(f"总样本数: {total_samples}")
        print(f"总掩码tokens: {masked_tokens_count}")
        print(f"平均掩码率: {masked_tokens_count / (total_samples * 50) if total_samples > 0 else 0:.2%}")
        print(f"输出文件: {output_file}")

class PretrainDataset(Dataset):
    """预训练数据集类"""
    
    def __init__(self, data_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载预训练数据
        self.data = []
        print(f"加载预训练数据: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        
        print(f"加载了 {len(self.data)} 个预训练样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取掩码后的文本和标签
        masked_text = item['input_text']
        labels = item['labels']
        
        # 使用tokenizer编码
        encoding = self.tokenizer(
            masked_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        # 处理标签，确保长度匹配
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        else:
            labels = labels + [-100] * (self.max_length - len(labels))
        
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_tensor,
            'original_label': torch.tensor(item.get('original_label', -1), dtype=torch.long)
        }

def create_pretrain_dataset(config):
    """创建预训练数据集的主函数"""
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 添加特殊tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    # 添加自定义tokens（与原始data_loader.py中相同的逻辑）
    custom_tokens = []
    
    # 时间相关tokens
    for year in range(2020, 2025):
        custom_tokens.append(f"YEAR_{year}")
    for month in range(1, 13):
        custom_tokens.append(f"MONTH_{month}")
    for day in range(1, 32):
        custom_tokens.append(f"DAY_{day}")
    for hour in range(0, 24):
        custom_tokens.append(f"HOUR_{hour}")
    for minute in range(0, 60):
        custom_tokens.append(f"MINUTE_{minute}")
    for second in range(0, 60):
        custom_tokens.append(f"SECOND_{second}")
    
    # 速度、航向、船舶类别tokens
    custom_tokens.extend(["SPD_STOP", "SPD_SLOW", "SPD_MID", "SPD_FAST", "SPD_HIGH"])
    custom_tokens.extend(["COG_N", "COG_NE", "COG_E", "COG_SE", "COG_S", "COG_SW", "COG_W", "COG_NW", "COG_UNKNOWN"])
    custom_tokens.extend([
        'Bulk_Carrier','Cargo_Ship','Container_Ship','Barge',
        'Fishing_Vessel','Other','Oil_Tanker','Passenger_Ship',
        'Sand_Carrier','Fishery_Research_Vessel','Supply_Ship',
        'Storage_Tanker', 'Submarine', 'Transport_Ship'
    ])
    custom_tokens.extend(["POINT_END"])
    
    # 收集H3 tokens
    print("收集H3 tokens...")
    h3_tokens = set()
    with open(config.train_data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num % 1000 == 0:
                print(f"处理第 {line_num} 行...")
            item = json.loads(line.strip())
            tokens = item['text'].split()
            for token in tokens:
                if token.startswith(('H3_', 'h3_', 'GRID_')):
                    h3_tokens.add(token)
    
    custom_tokens.extend(sorted(h3_tokens))
    print(f"收集到 {len(h3_tokens)} 个H3 tokens")
    
    # 添加tokens到tokenizer
    tokenizer.add_tokens(custom_tokens)
    print(f"Tokenizer词汇表大小: {len(tokenizer)}")
    
    # 创建数据处理器
    processor = PretrainDataProcessor(tokenizer)
    
    # 生成预训练数据
    pretrain_data_file = config.train_data_file.replace('.jsonl', '_pretrain.jsonl')
    processor.create_pretrain_data(
        config.train_data_file, 
        pretrain_data_file, 
        mask_prob=0.15
    )
    
    return pretrain_data_file, tokenizer

if __name__ == "__main__":
    # 测试代码
    from train_config import TrainingConfig
    
    config = TrainingConfig()
    pretrain_file, tokenizer = create_pretrain_dataset(config)
    
    # 创建数据集实例进行测试
    dataset = PretrainDataset(pretrain_file, tokenizer)
    
    # 查看一个样本
    sample = dataset[0]
    print("样本形状:")
    print(f"  input_ids: {sample['input_ids'].shape}")
    print(f"  attention_mask: {sample['attention_mask'].shape}")
    print(f"  labels: {sample['labels'].shape}")
    print(f"  掩码位置数: {(sample['labels'] != -100).sum().item()}")