import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

# NPU预训练配置参数
class PretrainConfigNPU:
    """NPU预训练阶段的配置参数"""
    
    # 模型配置
    model_name = "/home/maomao/pretrained_model/qwen2.5-0.5b"  # 基础模型路径
    
    # 数据配置
    train_data_file = "./train_dataset_demo.jsonl"  # 原始训练数据
    pretrain_data_file = "./train_dataset_demo_pretrain.jsonl"  # 生成的预训练数据
    
    max_length = 512  # 最大序列长度
    
    # MLM配置
    mask_prob = 0.15  # 总体掩码概率
    h3_mask_prob = 0.4  # H3地理位置掩码概率
    class_mask_prob = 0.3  # 船舶类别掩码概率
    motion_mask_prob = 0.2  # 速度/航向掩码概率
    random_mask_prob = 0.1  # 随机掩码概率
    
    # 训练参数（NPU优化）
    batch_size = 8  # NPU可以支持更大的batch size
    learning_rate = 5e-5  # 预训练用较大的学习率
    num_epochs = 5  # 预训练轮数
    warmup_steps = 1000  # 预热步数
    weight_decay = 0.01  # 权重衰减
    
    # 优化器配置
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-8
    max_grad_norm = 1.0  # 梯度裁剪
    
    # NPU特有配置
    device_type = "npu"  # 设备类型
    npu_ids = list(range(8))  # 使用的NPU ID列表 [0,1,2,3,4,5,6,7]
    master_addr = "127.0.0.1"  # 分布式训练主节点地址
    master_port = "29500"  # 分布式训练端口
    
    # 保存和日志配置
    output_dir = "../model_out/trajectory_pretrain_model_npu"  # NPU预训练模型保存路径
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
    
    # NPU性能优化
    use_amp = True  # 使用自动混合精度
    loss_scale = 128.0  # 损失缩放
    dataloader_num_workers = 4
    
    # 早停配置
    early_stopping_patience = 5  # 早停耐心值
    early_stopping_threshold = 0.001  # 改善阈值

class FineTuneConfigNPU:
    """NPU微调阶段的配置参数"""
    
    # 模型配置
    pretrained_model_path = "../model_out/trajectory_pretrain_model_npu/final_model"  # NPU预训练模型路径
    num_labels = 14  # 分类类别数
    
    # 数据配置  
    train_data_file = "./train_dataset_demo.jsonl"
    max_length = 512
    
    # 训练参数（微调用较小的学习率和较少的轮数）
    batch_size = 8  # NPU微调用较大的batch size
    learning_rate = 2e-5  # 微调用较小的学习率
    num_epochs = 10  # 微调轮数较少
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
    
    # NPU特有配置
    device_type = "npu"  # 设备类型
    npu_ids = list(range(8))  # 使用的NPU ID列表
    master_addr = "127.0.0.1"
    master_port = "29501"  # 使用不同端口避免冲突
    
    # 保存和日志配置
    output_dir = "../model_out/trajectory_finetune_model_npu"  # NPU微调模型保存路径
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
    
    # NPU性能优化
    use_amp = True  # 使用自动混合精度
    loss_scale = 128.0
    dataloader_num_workers = 4
    
    # 早停配置
    early_stopping_patience = 3  # 微调阶段更短的耐心值
    early_stopping_threshold = 0.001

class InferenceConfigNPU:
    """NPU推理阶段的配置参数"""
    
    # 模型配置
    model_path = "../model_out/trajectory_finetune_model_npu"  # NPU最终模型路径
    
    # 数据配置
    test_data_file = "../datasets/test_dataset_nj.jsonl"  # 测试数据
    max_length = 1024
    batch_size = 32  # NPU推理可以用较大的batch size
    
    # 推理配置
    device_type = "npu"
    npu_id = 0  # 推理使用单个NPU
    num_workers = 4
    
    # 输出配置
    output_file = "../results/inference_results_npu.json"
    save_predictions = True
    save_probabilities = True

# NPU配置管理器
class ConfigManagerNPU:
    """NPU配置管理器，用于统一管理不同阶段的配置"""
    
    @staticmethod
    def get_pretrain_config():
        """获取NPU预训练配置"""
        return PretrainConfigNPU()
    
    @staticmethod
    def get_finetune_config():
        """获取NPU微调配置"""
        return FineTuneConfigNPU()
    
    @staticmethod
    def get_inference_config():
        """获取NPU推理配置"""
        return InferenceConfigNPU()
    
    @staticmethod
    def setup_npu_environment():
        """设置NPU环境"""
        # 设置NPU设备
        if torch_npu.npu.is_available():
            torch.npu.set_device(0)  # 设置默认NPU设备
            print(f"NPU设备数量: {torch_npu.npu.device_count()}")
            for i in range(torch_npu.npu.device_count()):
                print(f"NPU {i}: {torch_npu.npu.get_device_name(i)}")
        else:
            raise RuntimeError("NPU设备不可用，请检查驱动和环境配置")
    
    @staticmethod
    def print_config(config, stage_name):
        """打印配置信息"""
        print(f"\n=== {stage_name} NPU配置 ===")
        for attr_name in dir(config):
            if not attr_name.startswith('_'):
                attr_value = getattr(config, attr_name)
                if not callable(attr_value):
                    print(f"{attr_name}: {attr_value}")
        print("=" * (len(stage_name) + 12))

if __name__ == "__main__":
    import torch_npu
    
    # 测试NPU配置
    try:
        config_manager = ConfigManagerNPU()
        config_manager.setup_npu_environment()
        
        pretrain_config = config_manager.get_pretrain_config()
        config_manager.print_config(pretrain_config, "预训练阶段")
        
        finetune_config = config_manager.get_finetune_config()
        config_manager.print_config(finetune_config, "微调阶段")
        
        inference_config = config_manager.get_inference_config()
        config_manager.print_config(inference_config, "推理阶段")
        
    except Exception as e:
        print(f"NPU配置测试失败: {e}")