# 预训练模型架构
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class TrajectoryPretrainModel(nn.Module):
    """
    轨迹预训练模型 - 使用MLM(Masked Language Modeling)任务
    
    该模型在预训练阶段学习：
    1. H3地理位置的空间连续性
    2. 船舶类别与轨迹行为的关联性  
    3. 速度、航向等运动特征的时序模式
    """
    
    def __init__(self, model_name, tokenizer=None, dropout_rate=0.1):
        super().__init__()
        
        # 加载基础backbone模型
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # 如果提供了tokenizer，调整embedding层大小以适应新的词汇表
        if tokenizer is not None:
            original_vocab_size = self.backbone.config.vocab_size
            new_vocab_size = len(tokenizer)
            
            if new_vocab_size != original_vocab_size:
                self.backbone.resize_token_embeddings(new_vocab_size)
                print(f"调整embedding层大小：{original_vocab_size} -> {new_vocab_size}")
        
        # 获取隐藏层维度
        self.hidden_size = self.backbone.config.hidden_size
        self.vocab_size = self.backbone.config.vocab_size if tokenizer is None else len(tokenizer)
        
        # MLM预测头
        self.mlm_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.vocab_size)
        )
        
        # 初始化MLM头的权重
        self._init_mlm_head()
    
    def _init_mlm_head(self):
        """初始化MLM头的权重"""
        for module in self.mlm_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ids, shape: (batch_size, seq_len)
            attention_mask: 注意力掩码, shape: (batch_size, seq_len)  
            labels: MLM标签, shape: (batch_size, seq_len), -100表示不计算loss
            
        Returns:
            dict: 包含loss和logits的字典
        """
        # 通过backbone获取隐藏状态
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 获取序列的隐藏状态
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # 通过MLM头预测被掩码的tokens
        mlm_logits = self.mlm_head(hidden_states)  # (batch_size, seq_len, vocab_size)
        
        outputs_dict = {"logits": mlm_logits}
        
        # 如果提供了标签，计算MLM损失
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 将logits和labels reshape为2D进行损失计算
            mlm_loss = loss_fct(
                mlm_logits.view(-1, self.vocab_size), 
                labels.view(-1)
            )
            outputs_dict["loss"] = mlm_loss
            
            # 计算准确率（只考虑被掩码的tokens）
            with torch.no_grad():
                predictions = torch.argmax(mlm_logits, dim=-1)
                mask = (labels != -100)
                if mask.sum() > 0:
                    accuracy = (predictions == labels)[mask].float().mean()
                    outputs_dict["accuracy"] = accuracy
                else:
                    outputs_dict["accuracy"] = torch.tensor(0.0, device=mlm_logits.device)
        
        return outputs_dict
    
    def save_pretrained(self, save_path):
        """保存预训练模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
        
        # 保存backbone的配置
        self.backbone.config.save_pretrained(save_path)
        
        print(f"预训练模型已保存到: {save_path}")
    
    @classmethod
    def from_pretrained(cls, pretrained_path, tokenizer=None):
        """从预训练路径加载模型"""
        import os
        from transformers import AutoConfig
        
        # 加载配置
        config = AutoConfig.from_pretrained(pretrained_path)
        
        # 创建模型实例
        model = cls(pretrained_path, tokenizer)
        
        # 加载权重
        model_path = os.path.join(pretrained_path, 'pytorch_model.bin')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"已从 {pretrained_path} 加载预训练权重")
        else:
            print(f"未找到预训练权重文件: {model_path}")
        
        return model

class TrajectoryFineTuneModel(nn.Module):
    """
    轨迹微调模型 - 用于船舶分类任务
    
    该模型加载预训练的backbone，添加分类头进行微调
    """
    
    def __init__(self, pretrained_model_path, num_labels, tokenizer=None, dropout_rate=0.3):
        super().__init__()
        
        # 选择加载方式
        if isinstance(pretrained_model_path, str) and pretrained_model_path.endswith(('.bin', '.pt')):
            # 如果是权重文件，需要先创建backbone再加载
            raise ValueError("请提供模型目录路径，而不是权重文件路径")
        else:
            # 加载预训练的backbone
            try:
                # 尝试加载我们的预训练模型
                pretrain_model = TrajectoryPretrainModel.from_pretrained(pretrained_model_path, tokenizer)
                self.backbone = pretrain_model.backbone
                print(f"成功加载预训练backbone: {pretrained_model_path}")
            except:
                # 如果失败，直接加载transformers模型
                self.backbone = AutoModel.from_pretrained(pretrained_model_path)
                if tokenizer is not None:
                    self.backbone.resize_token_embeddings(len(tokenizer))
                print(f"加载标准transformer模型: {pretrained_model_path}")
        
        # 获取隐藏层维度
        self.hidden_size = self.backbone.config.hidden_size
        
        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # 初始化分类头权重
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            labels: 分类标签（可选）
            
        Returns:
            logits或loss和logits
        """
        # 通过backbone获取隐藏状态
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 使用[CLS] token的表示进行分类
        pooled_output = outputs.last_hidden_state[:, 0]  # 取第一个token ([CLS])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
            # 计算准确率
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == labels).float().mean()
            
            return {
                "loss": loss,
                "logits": logits,
                "accuracy": accuracy
            }
        
        return {"logits": logits}
    
    def freeze_backbone(self, freeze=True):
        """冻结或解冻backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        if freeze:
            print("Backbone参数已冻结")
        else:
            print("Backbone参数已解冻")

class TrajectoryModelUtils:
    """轨迹模型的工具类"""
    
    @staticmethod
    def count_parameters(model):
        """统计模型参数数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
        print(f"冻结参数数: {total_params - trainable_params:,}")
        
        return total_params, trainable_params
    
    @staticmethod
    def get_model_size_mb(model):
        """获取模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        print(f"模型大小: {size_mb:.2f} MB")
        return size_mb

def test_models():
    """测试模型创建和基本功能"""
    print("=== 测试预训练模型 ===")
    
    # 创建预训练模型
    model_name = "/home/maomao/pretrained_model/qwen2.5-0.5b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    pretrain_model = TrajectoryPretrainModel(model_name, tokenizer)
    TrajectoryModelUtils.count_parameters(pretrain_model)
    TrajectoryModelUtils.get_model_size_mb(pretrain_model)
    
    # 测试前向传播
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, len(tokenizer), (batch_size, seq_len))
    labels[0, :5] = -100  # 部分位置不计算loss
    
    outputs = pretrain_model(input_ids, attention_mask, labels)
    print(f"MLM Loss: {outputs['loss'].item():.4f}")
    print(f"MLM Accuracy: {outputs['accuracy'].item():.4f}")
    
    print("\n=== 测试微调模型 ===")
    
    # 创建微调模型
    finetune_model = TrajectoryFineTuneModel(model_name, num_labels=14, tokenizer=tokenizer)
    TrajectoryModelUtils.count_parameters(finetune_model)
    
    # 测试分类
    class_labels = torch.randint(0, 14, (batch_size,))
    outputs = finetune_model(input_ids, attention_mask, class_labels)
    print(f"Classification Loss: {outputs['loss'].item():.4f}")
    print(f"Classification Accuracy: {outputs['accuracy'].item():.4f}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_models()