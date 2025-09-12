# 华为昇腾NPU轨迹分类推理脚本
import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import json
import numpy as np
from typing import List, Union, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import argparse

from pretrain_config_npu import InferenceConfigNPU

class TrajectoryInferenceNPU:
    """基于华为昇腾NPU的轨迹分类推理类"""
    
    def __init__(self, model_path: str, npu_id: int = 0):
        """
        初始化NPU推理引擎
        
        Args:
            model_path: 微调后的模型路径
            npu_id: 使用的NPU设备ID
        """
        self.model_path = model_path
        self.npu_id = npu_id
        
        # 设置NPU设备
        if torch_npu.npu.is_available():
            torch.npu.set_device(npu_id)
            self.device = torch.device(f'npu:{npu_id}')
            print(f"使用NPU设备: {self.device}")
            print(f"NPU设备名称: {torch_npu.npu.get_device_name(npu_id)}")
        else:
            raise RuntimeError("NPU设备不可用，请检查驱动和环境配置")
        
        self.tokenizer = None
        self.model = None
        self.class_names = None
        
        self._load_model()
    
    def _setup_npu_environment(self):
        """设置NPU推理环境"""
        # 设置NPU推理优化参数
        os.environ['TASK_QUEUE_ENABLE'] = '1'
        os.environ['PTCOPY_ENABLE'] = '1'
        os.environ['COMBINED_ENABLE'] = '1'
        os.environ['ACL_DUMP_DATA'] = '0'  # 推理时关闭dump
        
        # 清理NPU缓存
        torch_npu.npu.empty_cache()
    
    def _load_model(self):
        """加载模型和tokenizer"""
        print(f"从 {self.model_path} 加载模型...")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print(f"Tokenizer词汇表大小: {len(self.tokenizer)}")
            
            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # 获取类别数
            self.num_labels = self.model.config.num_labels
            self.class_names = [f"Class_{i}" for i in range(self.num_labels)]
            
            print(f"模型加载成功，类别数: {self.num_labels}")
            
            # 设置NPU推理优化
            self._setup_npu_environment()
            
            # 模型预热
            self._warmup_model()
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def _warmup_model(self):
        """模型预热，提高推理速度"""
        print("NPU模型预热中...")
        
        # 创建dummy输入进行预热
        dummy_text = "YEAR_2021 MONTH_9 DAY_17 HOUR_6 MINUTE_11 SECOND_42 " \
                    "H3_CHAR_0_8 H3_CHAR_1_a SPD_STOP COG_NE Bulk_Carrier POINT_END"
        
        # 预热几次
        for _ in range(3):
            with torch.no_grad():
                inputs = self.tokenizer(
                    dummy_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding='max_length'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                _ = self.model(**inputs)
        
        torch_npu.npu.synchronize()
        print("NPU模型预热完成")
    
    def predict_single(self, text: str, return_probs: bool = False) -> Union[int, Dict]:
        """
        单个样本预测
        
        Args:
            text: 输入的轨迹文本
            return_probs: 是否返回概率分布
            
        Returns:
            预测类别ID或包含类别和概率的字典
        """
        with torch.no_grad():
            # Tokenize输入
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding='max_length'
            )
            
            # 移动到NPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 获取预测结果
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][predicted_class].item()
            
            if return_probs:
                return {
                    'predicted_class': predicted_class,
                    'class_name': self.class_names[predicted_class],
                    'confidence': confidence,
                    'probabilities': probs[0].cpu().numpy().tolist()
                }
            else:
                return predicted_class
    
    def predict_batch(self, texts: List[str], batch_size: int = 32, 
                     return_probs: bool = False) -> List[Union[int, Dict]]:
        """
        批量预测
        
        Args:
            texts: 输入的轨迹文本列表
            batch_size: 批次大小
            return_probs: 是否返回概率分布
            
        Returns:
            预测结果列表
        """
        results = []
        
        print(f"批量推理 {len(texts)} 个样本，批次大小: {batch_size}")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="NPU批量推理"):
                batch_texts = texts[i:i+batch_size]
                
                # Batch tokenization
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding='max_length'
                )
                
                # 移动到NPU
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 批量推理
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 处理结果
                probs = torch.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(logits, dim=-1)
                
                for j in range(len(batch_texts)):
                    predicted_class = predicted_classes[j].item()
                    confidence = probs[j][predicted_class].item()
                    
                    if return_probs:
                        results.append({
                            'predicted_class': predicted_class,
                            'class_name': self.class_names[predicted_class],
                            'confidence': confidence,
                            'probabilities': probs[j].cpu().numpy().tolist()
                        })
                    else:
                        results.append(predicted_class)
        
        return results
    
    def predict_from_file(self, input_file: str, output_file: str = None, 
                         batch_size: int = 32) -> List[Dict]:
        """
        从文件读取数据进行预测
        
        Args:
            input_file: 输入JSONL文件路径
            output_file: 输出文件路径（可选）
            batch_size: 批次大小
            
        Returns:
            预测结果列表
        """
        print(f"从文件 {input_file} 读取数据...")
        
        # 读取数据
        texts = []
        original_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                texts.append(item['text'])
                original_data.append(item)
        
        print(f"读取到 {len(texts)} 个样本")
        
        # 批量预测
        predictions = self.predict_batch(texts, batch_size, return_probs=True)
        
        # 组合结果
        results = []
        for i, (original, prediction) in enumerate(zip(original_data, predictions)):
            result = {
                'id': i,
                'original_text': original['text'],
                'true_label': original.get('label', -1),
                'predicted_class': prediction['predicted_class'],
                'class_name': prediction['class_name'],
                'confidence': prediction['confidence'],
                'probabilities': prediction['probabilities']
            }
            results.append(result)
        
        # 保存结果
        if output_file:
            print(f"保存预测结果到 {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        return results
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'npu_device_name': torch_npu.npu.get_device_name(self.npu_id),
            'num_labels': self.num_labels,
            'vocab_size': len(self.tokenizer),
            'max_length': self.tokenizer.model_max_length
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='华为昇腾NPU轨迹分类推理')
    parser.add_argument('--model_path', type=str, required=True,
                        help='微调后的模型路径')
    parser.add_argument('--input_file', type=str, default=None,
                        help='输入JSONL文件路径')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出文件路径')
    parser.add_argument('--text', type=str, default=None,
                        help='单个文本输入')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--npu_id', type=int, default=0,
                        help='使用的NPU设备ID')
    
    args = parser.parse_args()
    
    try:
        # 创建推理引擎
        inferencer = TrajectoryInferenceNPU(args.model_path, args.npu_id)
        
        # 显示模型信息
        model_info = inferencer.get_model_info()
        print("\n=== 模型信息 ===")
        for key, value in model_info.items():
            print(f"{key}: {value}")
        print("=" * 20)
        
        if args.text:
            # 单个文本预测
            print(f"\n单个文本预测:")
            print(f"输入: {args.text}")
            
            result = inferencer.predict_single(args.text, return_probs=True)
            print(f"预测类别: {result['predicted_class']}")
            print(f"类别名称: {result['class_name']}")
            print(f"置信度: {result['confidence']:.4f}")
            
        elif args.input_file:
            # 文件批量预测
            print(f"\n从文件批量预测:")
            results = inferencer.predict_from_file(
                args.input_file, 
                args.output_file, 
                args.batch_size
            )
            
            # 计算准确率（如果有真实标签）
            if results and results[0]['true_label'] != -1:
                correct = sum(1 for r in results 
                            if r['predicted_class'] == r['true_label'])
                accuracy = correct / len(results)
                print(f"准确率: {accuracy:.4f} ({correct}/{len(results)})")
        
        else:
            print("请提供 --text 或 --input_file 参数")
    
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()