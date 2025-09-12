import torch
# 预训练配置参数
class PretrainConfig:
    """预训练阶段的配置参数"""
    
    # 模型配置
    model_name = "/home/maomao/pretrained_model/qwen2.5-0.5b"  # 基础模型路径
    
    # 数据配置
    # train_data_file = "../datasets/train_dataset_nj.jsonl"  # 原始训练数据
    # pretrain_data_file = "../datasets/train_dataset_nj_pretrain.jsonl"  # 生成的预训练数据
    train_data_file = "./train_dataset_demo.jsonl"  # 原始训练数据
    pretrain_data_file = "./train_dataset_demo_pretrain.jsonl"  # 生成的预训练数据
    
    
    max_length = 512  # 最大序列长度
    
    # MLM配置
    mask_prob = 0.15  # 总体掩码概率
    h3_mask_prob = 0.4  # H3地理位置掩码概率
    class_mask_prob = 0.3  # 船舶类别掩码概率
    motion_mask_prob = 0.2  # 速度/航向掩码概率
    random_mask_prob = 0.1  # 随机掩码概率
    
    # 训练参数
    batch_size = 4  # 预训练可以用较大的batch size
    learning_rate = 5e-5  # 预训练用较大的学习率
    num_epochs = 5  # 预训练轮数
    warmup_steps = 1000  # 预热步数
    weight_decay = 0.01  # 权重衰减
    
    # 优化器配置
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-8
    max_grad_norm = 1.0  # 梯度裁剪
    
    # 保存和日志配置
    output_dir = "../model_out/trajectory_pretrain_model"  # 预训练模型保存路径
    logging_steps = 100  # 日志打印间隔
    save_steps = 2000  # 模型保存间隔
    eval_steps = 1000  # 验证间隔
    save_total_limit = 3  # 最多保存的检查点数
    
    # 验证配置
    eval_ratio = 0.1  # 验证集比例
    
    # 分布式训练配置
    local_rank = -1
    find_unused_parameters = True
    gradient_accumulation_steps = 1
    fp16 = True  # 混合精度训练
    dataloader_num_workers = 4
    
    # 早停配置
    early_stopping_patience = 5  # 早停耐心值
    early_stopping_threshold = 0.001  # 改善阈值

class FineTuneConfig:
    """微调阶段的配置参数"""
    
    # 模型配置
    pretrained_model_path = "../model_out/trajectory_pretrain_model/final_model"  # 预训练模型路径
    num_labels = 14  # 分类类别数
    
    # 数据配置  
    train_data_file = "./train_dataset_demo.jsonl"
    max_length = 512
    
    # 训练参数（微调用较小的学习率和较少的轮数）
    batch_size = 4  # 微调用较小的batch size
    learning_rate = 2e-5  # 微调用较小的学习率
    num_epochs = 5  # 微调轮数较少
    warmup_steps = 500
    weight_decay = 0.01
    
    # 微调特有配置
    freeze_backbone_epochs = 0  # 前几轮冻结backbone，0表示不冻结
    dropout_rate = 0.3  # 分类头的dropout率
    
    # 优化器配置
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    
    # 保存和日志配置
    output_dir = "../model_out/trajectory_finetune_model"  # 微调模型保存路径
    logging_steps = 50
    save_steps = 500
    eval_steps = 500
    save_total_limit = 2
    
    # 验证配置
    eval_ratio = 0.2  # 微调阶段用更多数据做验证
    
    # 分布式训练配置
    local_rank = -1
    find_unused_parameters = True
    gradient_accumulation_steps = 2  # 微调时可以用梯度累积
    fp16 = True
    dataloader_num_workers = 4
    
    # 早停配置
    early_stopping_patience = 3  # 微调阶段更短的耐心值
    early_stopping_threshold = 0.001

class InferenceConfig:
    """推理阶段的配置参数"""
    
    # 模型配置
    model_path = "../model_out/trajectory_finetune_model"  # 最终模型路径
    
    # 数据配置
    test_data_file = "../datasets/test_dataset_nj.jsonl"  # 测试数据
    max_length = 1024
    batch_size = 16  # 推理可以用较大的batch size
    
    # 推理配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    
    # 输出配置
    output_file = "../results/inference_results.json"
    save_predictions = True
    save_probabilities = True

# 全局配置管理器
class ConfigManager:
    """配置管理器，用于统一管理不同阶段的配置"""
    
    @staticmethod
    def get_pretrain_config():
        """获取预训练配置"""
        return PretrainConfig()
    
    @staticmethod
    def get_finetune_config():
        """获取微调配置"""
        return FineTuneConfig()
    
    @staticmethod
    def get_inference_config():
        """获取推理配置"""
        return InferenceConfig()
    
    @staticmethod
    def print_config(config, stage_name):
        """打印配置信息"""
        print(f"\n=== {stage_name} 配置 ===")
        for attr_name in dir(config):
            if not attr_name.startswith('_'):
                attr_value = getattr(config, attr_name)
                if not callable(attr_value):
                    print(f"{attr_name}: {attr_value}")
        print("=" * (len(stage_name) + 8))

if __name__ == "__main__":
    import torch
    
    # 测试配置
    config_manager = ConfigManager()
    
    pretrain_config = config_manager.get_pretrain_config()
    config_manager.print_config(pretrain_config, "预训练阶段")
    
    finetune_config = config_manager.get_finetune_config()
    config_manager.print_config(finetune_config, "微调阶段")
    
    inference_config = config_manager.get_inference_config()
    config_manager.print_config(inference_config, "推理阶段")