import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from train_config import TrainingConfig

class TrajectoryDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512, test_size=0.2):
        """
        轨迹分类数据集
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        self.data = []
        self.labels = []
        
        print(f"加载数据文件: {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item['text'])
                self.labels.append(item['label'])
        
        print(f"总共加载 {len(self.data)} 个训练样本")
        print(f"标签分布: {set(self.labels)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        
        # 使用tokenizer编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(config, is_distributed=False, rank=0, world_size=1):
    """创建训练和验证数据加载器，支持分布式训练"""
    
    # 初始化tokenizer（只在主进程打印信息）
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if rank == 0:
        print(f"使用的tokenizer: {config.model_name}")
        print(f"词汇表大小: {tokenizer.vocab_size}")
    
    # 添加pad_token如果不存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 添加自定义tokens到tokenizer词汇表
    custom_tokens = []
    
    # 时间相关tokens
    for year in range(2020, 2025):  # 支持2020-2024年
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
    
    # 速度tokens
    custom_tokens.extend(["SPD_STOP", "SPD_SLOW", "SPD_MID", "SPD_FAST", "SPD_HIGH"])
    
    # 航向tokens  
    custom_tokens.extend(["COG_N", "COG_NE", "COG_E", "COG_SE", "COG_S", "COG_SW", "COG_W", "COG_NW", "COG_UNKNOWN"])
    
    # 船舶类别tokens
    custom_tokens.extend([
        'Bulk_Carrier','Cargo_Ship','Container_Ship','Barge',
        'Fishing_Vessel','Other','Oil_Tanker','Passenger_Ship',
        'Sand_Carrier','Fishery_Research_Vessel','Supply_Ship',
        'Storage_Tanker', 'Submarine', 'Transport_Ship'
    ])
    
    # 结构化tokens
    custom_tokens.extend(["POINT_END"])
    
    # H3地理tokens - 这些会在运行时动态收集（只在主进程收集）
    if rank == 0:
        print("正在收集训练数据中的H3 tokens...")
        h3_tokens = set()
        with open(config.train_data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % 1000 == 0:
                    print(f"处理第 {line_num} 行...")
                item = json.loads(line.strip())
                tokens = item['text'].split()
                for token in tokens:
                    if token.startswith(('H3_', 'h3_')):
                        h3_tokens.add(token)
        
        custom_tokens.extend(sorted(h3_tokens))
        print(f"收集到 {len(h3_tokens)} 个H3 tokens")
    
    # 在分布式训练中，需要同步所有进程的token列表
    if is_distributed:
        # 广播token列表到所有进程
        import torch.distributed as dist
        if rank == 0:
            # 主进程将token列表序列化并广播
            token_str = json.dumps(custom_tokens)
            token_tensor = torch.tensor([len(token_str.encode('utf-8'))], dtype=torch.int64).cuda()
        else:
            token_tensor = torch.tensor([0], dtype=torch.int64).cuda()
        
        # 广播字符串长度
        dist.broadcast(token_tensor, 0)
        str_len = token_tensor.item()
        
        if rank == 0:
            token_bytes = token_str.encode('utf-8')
            padded_bytes = token_bytes + b'\0' * (str_len - len(token_bytes))
            token_tensor = torch.frombuffer(padded_bytes, dtype=torch.uint8).cuda()
        else:
            token_tensor = torch.zeros(str_len, dtype=torch.uint8).cuda()
        
        # 广播token数据
        dist.broadcast(token_tensor, 0)
        
        if rank != 0:
            token_bytes = token_tensor.cpu().numpy().tobytes()
            token_str = token_bytes.decode('utf-8').rstrip('\0')
            custom_tokens = json.loads(token_str)
    
    # 添加tokens到tokenizer
    if rank == 0:
        print(f"添加 {len(custom_tokens)} 个自定义tokens到tokenizer...")

    tokenizer.add_tokens(custom_tokens)
    
    if rank == 0:
        print(f"增加后Tokenizer词汇表大小: {len(tokenizer)}")
    
    # 创建数据集
    full_dataset = TrajectoryDataset(
        config.train_data_file, 
        tokenizer, 
        config.max_length
    )
    
    # 分割训练和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    if is_distributed:
        # 分布式训练使用DistributedSampler
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
        # 单卡训练
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
    
    if rank == 0:
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        if is_distributed:
            print(f"分布式训练: 每个GPU处理 ~{len(train_dataset)//world_size} 个训练样本")
    
    return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    # 测试数据加载器
    config = TrainingConfig()
    train_loader, val_loader, tokenizer = create_data_loaders(config)
    
    # 查看一个批次的数据
    for batch in train_loader:
        print("批次形状:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        print(f"  labels值: {batch['labels']}")
        break