# 训练配置参数
class TrainingConfig:
    # 模型配置
    model_name = "/home/maomao/pretrained_model/qwen2.5-0.5b"  # 使用本地下载的模型
    
    # 数据配置
    train_data_file = "../datasets/train_dataset_nj.jsonl"
    max_length = 1024  # 最大序列长度
    
    # 训练参数 (针对分布式训练调整)
    batch_size = 4      # 每个GPU的batch size，8卡总共64个样本/批次
    learning_rate = 2e-5
    num_epochs = 100
    warmup_steps = 500  # 增加warmup步数，适应分布式训练
    
    # 保存和日志
    output_dir = "../model_out/qwen_trajectory_model"
    logging_steps = 50   # 增加日志间隔，减少输出频率
    save_steps = 500
    
    # 分类任务配置
    num_labels = 14  # 根据你的船舶类别数调整
    
    # 分布式训练和优化配置
    gradient_accumulation_steps = 1  # 分布式训练时通常不需要梯度累积
    fp16 = True  # 使用混合精度训练，节省显存
    dataloader_num_workers = 4  # 增加数据加载器的工作进程数
    
    # 分布式训练特有配置
    local_rank = -1  # 将由启动脚本设置
    find_unused_parameters = True  # DDP参数，处理部分参数未使用的情况