import torch
import json
from transformers import AutoTokenizer
from train_model import TrajectoryClassifier
from train_config import TrainingConfig

class TrajectoryPredictor:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载tokenizer并添加自定义tokens（与训练时保持一致）
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 添加与训练时相同的自定义tokens
        self._add_custom_tokens()
        
        # 加载模型（传入tokenizer以正确调整embedding层大小）
        self.model = TrajectoryClassifier(config.model_name, config.num_labels, self.tokenizer)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成，使用设备: {self.device}")
    
    def _add_custom_tokens(self):
        """添加与训练时相同的自定义tokens"""
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
        
        # 速度tokens
        custom_tokens.extend(["SPD_STOP", "SPD_SLOW", "SPD_MID", "SPD_FAST", "SPD_HIGH"])
        
        # 航向tokens  
        custom_tokens.extend(["COG_N", "COG_NE", "COG_E", "COG_SE", "COG_S", "COG_SW", "COG_W", "COG_NW", "COG_UNKNOWN"])
        
        # 船舶类别tokens
        custom_tokens.extend([
            "CLASS_RESERVED", "CLASS_WIG", "CLASS_PASSENGER", "CLASS_CARGO", 
            "CLASS_TANKER", "CLASS_OTHER", "CLASS_HSC", "CLASS_PLEASURE", "CLASS_UNKNOWN"
        ])
        
        # 结构化tokens
        custom_tokens.extend(["POINT_END"])
        
        # H3地理tokens - 从训练数据中动态收集
        h3_tokens = set()
        with open(self.config.train_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                tokens = item['text'].split()
                for token in tokens:
                    if token.startswith(('H3_', 'h3_')):
                        h3_tokens.add(token)
        
        custom_tokens.extend(sorted(h3_tokens))
        
        # 添加tokens到tokenizer
        self.tokenizer.add_tokens(custom_tokens)
        print(f"添加了 {len(custom_tokens)} 个自定义tokens，tokenizer词汇表大小: {len(self.tokenizer)}")
    
    def predict(self, text):
        """预测单个轨迹的类别"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }
    
    def predict_batch(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

def test_model():
    """测试训练好的模型"""
    config = TrainingConfig()
    model_path = f"{config.output_dir}/best_model.pt"
    
    try:
        predictor = TrajectoryPredictor(model_path, config)
        
        # 从验证数据中取几个样本测试
        print("加载测试数据...")
        test_samples = []
        with open(config.train_data_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:  # 取前5个样本测试
                    data = json.loads(line)
                    test_samples.append({
                        'text': data['text'],
                        'true_label': data['label']
                    })
        
        print("\n测试结果:")
        for i, sample in enumerate(test_samples):
            result = predictor.predict(sample['text'])
            print(f"\n样本 {i+1}:")
            print(f"  真实标签: {sample['true_label']}")
            print(f"  预测标签: {result['predicted_class']}")
            print(f"  置信度: {result['confidence']:.4f}")
            print(f"  文本前100字符: {sample['text'][:100]}...")
    
    except FileNotFoundError:
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练脚本: python train_model.py")

if __name__ == "__main__":
    test_model()