# 昇腾NPU训练配置参数
class TrainingConfigNPU:
    # 模型配置
    model_name = "/home/maomao/qwen3-0.6b"  # 使用本地下载的模型
    
    # 数据配置
    train_data_file = "train_dataset_nj.jsonl"
    max_length = 512  # 昇腾NPU建议较小的序列长度
    
    # 训练参数 (针对昇腾NPU优化)
    batch_size = 6      # 每个NPU的batch size，8卡总共48个样本/批次
    learning_rate = 1e-5  # NPU推荐较小的学习率
    num_epochs = 100
    warmup_steps = 300   # 适应NPU的warmup设置
    
    # 保存和日志
    output_dir = "./qwen_trajectory_model_npu"
    logging_steps = 50
    save_steps = 500
    
    # 分类任务配置
    num_labels = 14  # 根据你的船舶类别数调整
    
    # NPU特有配置
    gradient_accumulation_steps = 1
    fp16 = True  # 昇腾NPU支持混合精度
    dataloader_num_workers = 2  # NPU环境建议较少的workers
    
    # NPU优化配置
    npu_loss_scale = 128.0  # NPU损失缩放因子
    npu_opt_level = "O1"    # NPU混合精度级别：O0(FP32), O1(推荐), O2(激进)
    
    # 分布式训练配置
    local_rank = -1
    find_unused_parameters = False  # NPU建议设置为False