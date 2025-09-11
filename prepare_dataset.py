# /home/maomao/project/token/prepare_dataset.py
import os
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime

# ====================================================================
# 1. 轨迹点令牌化函数
#    这些函数将每个轨迹点的数值数据转换为离散的文本令牌
# ====================================================================

def time_to_tokens(date_str):
    """将时间戳转换为具体时间组件令牌列表"""
    dt = pd.to_datetime(date_str)
    return [
        f"YEAR_{dt.year}",
        f"MONTH_{dt.month}",
        f"DAY_{dt.day}",
        f"HOUR_{dt.hour}",
        f"MINUTE_{dt.minute}",
        f"SECOND_{dt.second}"
    ]

def speed_to_token(sog):
    """将速度(SOG)转换为令牌"""
    if sog < 2: return "SPD_STOP"
    elif sog < 5: return "SPD_SLOW"
    elif sog < 10: return "SPD_MID"
    elif sog < 15: return "SPD_FAST"
    else: return "SPD_HIGH"

def course_to_token(cog):
    """将航向(COG)转换为精细化的八个方向令牌"""
    if pd.isna(cog): return "COG_UNKNOWN"
    cog = float(cog) % 360  # 确保在0-360范围内
    
    # 8个精细方向，每个方向45度
    if 0 <= cog < 22.5 or 337.5 <= cog < 360: return "COG_N"      # 北
    elif 22.5 <= cog < 67.5: return "COG_NE"    # 东北
    elif 67.5 <= cog < 112.5: return "COG_E"    # 东
    elif 112.5 <= cog < 157.5: return "COG_SE"  # 东南
    elif 157.5 <= cog < 202.5: return "COG_S"   # 南
    elif 202.5 <= cog < 247.5: return "COG_SW"  # 西南
    elif 247.5 <= cog < 292.5: return "COG_W"   # 西
    elif 292.5 <= cog < 337.5: return "COG_NW"  # 西北
    else: return "COG_UNKNOWN"

def h3_to_tokens_hierarchical_optimized(h3_str):
    """优化的层级化H3令牌，去掉H3_FULL以减少词表大小"""
    h3_str = str(h3_str).lower()
    
    if len(h3_str) >= 15:  # 标准H3长度
        tokens = []
        # 分辨率级别（固定范围0-15，词表大小固定）
        tokens.append(f"H3_RES_{h3_str[1]}")
        
        # 多层级地理区域（显著减少词表大小）
        tokens.append(f"H3_L1_{h3_str[:3]}")   # 大区域 (~16^2 = 256个)
        tokens.append(f"H3_L2_{h3_str[:5]}")   # 中区域 (~16^4 = 65k个)
        tokens.append(f"H3_L3_{h3_str[:7]}")   # 小区域 (~16^6 = 16M个，但可控)
        tokens.append(f"H3_L4_{h3_str[:9]}")   # 精细区域
        
        # 不再使用H3_FULL，避免词表爆炸
        return tokens
    else:
        return [f"H3_SHORT_{h3_str}"]

def h3_to_tokens_char_split(h3_str):
    """将H3字符串的每一个字符拆开作为独立token"""
    h3_str = str(h3_str).lower()
    tokens = []
    
    # 为每个字符创建一个token
    for i, char in enumerate(h3_str):
        tokens.append(f"H3_CHAR_{i}_{char}")
    
    return tokens

def h3_to_tokens_semantic(h3_str):
    """基于H3语义结构的令牌化，根据实际含义拆分"""
    h3_str = str(h3_str).lower()
    
    if len(h3_str) >= 15:
        tokens = []
        
        # 1. 版本信息（通常是8）
        tokens.append(f"H3_VER_{h3_str[0]}")
        
        # 2. 分辨率级别（0-15）
        resolution = h3_str[1]
        tokens.append(f"H3_RES_{resolution}")
        
        # 3. 基础单元编码（位置的粗粒度表示）
        base_unit = h3_str[2:6]  # 4位用于基础单元
        tokens.append(f"H3_BASE_{base_unit}")
        
        # 4. 子单元编码（位置的细粒度表示）
        sub_unit = h3_str[6:10]  # 4位用于子单元
        tokens.append(f"H3_SUB_{sub_unit}")
        
        # 5. 精确单元（最后几位）
        precise_unit = h3_str[10:13]  # 3位用于精确定位
        tokens.append(f"H3_PRECISE_{precise_unit}")
        
        return tokens
    else:
        return [f"H3_SHORT_{h3_str}"]

def h3_to_tokens_grid_based(h3_str):
    """基于网格概念的H3令牌化，更抽象的地理表示"""
    h3_str = str(h3_str).lower()
    
    if len(h3_str) >= 15:
        tokens = []
        
        # 分辨率
        resolution = h3_str[1]  # 保持为字符串
        tokens.append(f"GRID_RES_{resolution}")
        
        # 将H3转换为网格坐标概念
        # 大网格（全球划分）
        major_grid = h3_str[2:4]
        tokens.append(f"GRID_MAJOR_{major_grid}")
        
        # 中等网格（区域划分）
        mid_grid = h3_str[4:7]
        tokens.append(f"GRID_MID_{mid_grid}")
        
        # 小网格（局部划分）
        minor_grid = h3_str[7:10]
        tokens.append(f"GRID_MINOR_{minor_grid}")
        
        # 微网格（精确位置）
        micro_grid = h3_str[10:13]
        tokens.append(f"GRID_MICRO_{micro_grid}")
        
        return tokens
    else:
        return [f"GRID_SHORT_{h3_str}"]

def h3_to_tokens_relative(h3_str, prev_h3_str=None):
    """基于相对位置的H3令牌化，减少绝对位置依赖"""
    h3_str = str(h3_str).lower()
    
    if prev_h3_str is None:
        # 第一个点，使用简化的绝对位置
        return h3_to_tokens_hierarchical_optimized(h3_str)
    
    prev_h3_str = str(prev_h3_str).lower()
    tokens = []
    
    # 比较分辨率变化
    if len(h3_str) >= 2 and len(prev_h3_str) >= 2:
        curr_res = h3_str[1]
        prev_res = prev_h3_str[1] 
        
        if curr_res == prev_res:
            tokens.append("H3_RES_SAME")
        else:
            tokens.append(f"H3_RES_CHANGE_{curr_res}")
    
    # 比较区域变化（逐层比较）
    for i, (level_name, end_pos) in enumerate([("L1", 3), ("L2", 5), ("L3", 7), ("L4", 9)]):
        if len(h3_str) >= end_pos and len(prev_h3_str) >= end_pos:
            curr_level = h3_str[:end_pos]
            prev_level = prev_h3_str[:end_pos]
            
            if curr_level == prev_level:
                tokens.append(f"H3_{level_name}_SAME")
            else:
                tokens.append(f"H3_{level_name}_CHANGE")
                # 只在变化时记录新的区域（减少词表）
                tokens.append(f"H3_{level_name}_NEW_{curr_level[-2:]}")  # 只记录最后2位
    
    return tokens

# 保留原函数以保持兼容性
def h3_to_tokens_hierarchical(h3_str):
    """将H3索引转换为层级化令牌，保留地理层级信息（已优化）"""
    return h3_to_tokens_hierarchical_optimized(h3_str)




def vessel_class_to_token(vessel_class):
    """将船舶类别转换为令牌"""
    class_map = {
        0: 'Bulk_Carrier',
        1: 'Cargo_Ship',
        2: 'Container_Ship',
        3: 'Barge',
        4: 'Fishing_Vessel',
        5: 'Other',
        6: 'Oil_Tanker',
        7: 'Passenger_Ship',
        8: 'Sand_Carrier',
        9: 'Fishery_Research_Vessel',
        10: 'Supply_Ship',
        11: 'Storage_Tanker',
        12: 'Submarine',
        13: 'Transport_Ship'
    }
    return class_map.get(int(vessel_class), "CLASS_UNKNOWN")

# ====================================================================
# H3 tokenization配置
# ====================================================================
H3_TOKENIZATION_OPTIONS = {
    'char_split': h3_to_tokens_char_split,                           # 最简单：每个字符单独token
    'hierarchical_optimized': h3_to_tokens_hierarchical_optimized,  # 推荐：去掉H3_FULL的层级方案
    'semantic': h3_to_tokens_semantic,                            # 基于H3语义结构
    'grid_based': h3_to_tokens_grid_based,                       # 基于网格概念
    'relative': h3_to_tokens_relative,                           # 相对位置编码
    'original_hierarchical': h3_to_tokens_hierarchical,          # 原层级方案（已优化）
}

def get_h3_tokenizer(method='hierarchical_optimized'):
    """获取指定的H3令牌化函数"""
    return H3_TOKENIZATION_OPTIONS.get(method, h3_to_tokens_hierarchical_optimized)

def estimate_vocab_size(method='hierarchical_optimized'):
    """估算不同方案的词表大小"""
    estimates = {
        'char_split': {
            'H3_CHAR_0_*': 1,           # 第0位通常是8
            'H3_CHAR_1_*': 16,          # 第1位是0-f (分辨率)
            'H3_CHAR_2_*': 16,          # 第2位是0-f
            'H3_CHAR_3_*': 16,          # 第3位是0-f
            # ... 总共15个位置，每个位置16个可能值
            'total_per_position': 16,
            'total_positions': 15,      # H3通常15位
            'total_estimate': '15 × 16 = 240个token'
        },
        'hierarchical_optimized': {
            'H3_RES': 16,           # 分辨率0-15
            'H3_L1': 256,           # ~16^2
            'H3_L2': 65536,         # ~16^4  
            'H3_L3': 16777216,      # ~16^6 (但实际会少很多)
            'H3_L4': 'variable',    # 取决于数据分布
            'total_estimate': '< 20M'
        },
        'semantic': {
            'H3_VER': 2,            # 通常只有版本8
            'H3_RES': 16,           # 分辨率0-15
            'H3_BASE': 65536,       # 4位hex = 16^4
            'H3_SUB': 65536,        # 4位hex = 16^4  
            'H3_PRECISE': 4096,     # 3位hex = 16^3
            'total_estimate': '< 150K'
        },
        'grid_based': {
            'GRID_RES': 16,         # 分辨率0-15
            'GRID_MAJOR': 256,      # 2位hex = 16^2
            'GRID_MID': 4096,       # 3位hex = 16^3
            'GRID_MINOR': 4096,     # 3位hex = 16^3
            'GRID_MICRO': 4096,     # 3位hex = 16^3
            'total_estimate': '< 13K'
        },
        'relative': {
            'description': '词表大小取决于数据中的实际变化模式，通常很小',
            'total_estimate': '< 1K'
        }
    }
    return estimates.get(method, '未知')

# ====================================================================
# 2. 核心处理函数，将单个CSV文件中的每个轨迹点转换为令牌序列
# ====================================================================
def process_track_file_to_segments(csv_path, segment_size=30, h3_method='hierarchical_optimized'):
    """读取单个CSV文件，将轨迹按指定大小分段，返回分段token序列和标签列表"""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 获取轨迹的类别标签（假设整个轨迹的类别相同）
    vessel_class = df['class'].iloc[0] if len(df) > 0 else 0
    
    # 获取H3令牌化函数
    h3_tokenizer = get_h3_tokenizer(h3_method)
    
    segments = []
    
    # 按segment_size个点分段处理
    for start_idx in range(0, len(df), segment_size):
        end_idx = min(start_idx + segment_size, len(df))
        segment_df = df.iloc[start_idx:end_idx]
        
        # 如果分段点数太少，跳过（比如少于10个点）
        if len(segment_df) < 10:
            continue
            
        segment_tokens = []
        prev_h3 = None  # 用于相对位置编码
        
        for _, row in segment_df.iterrows():
            # 为每个轨迹点生成令牌序列
            point_tokens = []
            
            # 1. 时间令牌（拆分为具体组件）
            time_tokens = time_to_tokens(row['date'])
            point_tokens.extend(time_tokens)
            
            # 2. H3位置令牌（使用选定的方案）
            if h3_method == 'relative':
                h3_tokens = h3_tokenizer(row['H3'], prev_h3)
                prev_h3 = row['H3']  # 更新前一个H3值
            else:
                h3_tokens = h3_tokenizer(row['H3'])
            point_tokens.extend(h3_tokens)
            
            # 3. 速度令牌
            point_tokens.append(speed_to_token(row['sog']))
            
            # 4. 航向令牌
            point_tokens.append(course_to_token(row['cog']))
            
            # 5. 船舶类别令牌
            point_tokens.append(vessel_class_to_token(row['class']))
            
            # 添加点结束标记
            point_tokens.append("POINT_END")
            segment_tokens.extend(point_tokens)
        
        # 将此分段的所有令牌连接成字符串
        segment_text = " ".join(segment_tokens)
        segments.append({
            "text": segment_text,
            "label": int(vessel_class),
            "segment_length": len(segment_df),
            "h3_method": h3_method  # 记录使用的H3方案
        })
    
    return segments

# ====================================================================
# 3. 主程序：遍历数据文件夹，生成JSONL格式的数据集
# ====================================================================
def create_dataset(input_folder, output_file, segment_size=30, h3_method='hierarchical_optimized'):
    """
    处理文件夹中的所有CSV文件，将它们按分段处理并保存为JSONL格式。
    每个分段包含指定数量的轨迹点和对应的类别标签。
    
    参数:
    - h3_method: H3令牌化方案，可选值：
      * 'char_split': 每个字符单独作为token (最小词表：240个token)
      * 'hierarchical_optimized' (推荐): 优化的层级方案，去掉H3_FULL
      * 'semantic': 基于H3语义结构的拆分
      * 'grid_based': 基于网格概念的表示  
      * 'relative': 相对位置编码
      * 'grouped': 分组方案
      * 'simple': 简单方案
    """
    print(f"开始处理文件夹: {input_folder}")
    print(f"分段大小: {segment_size}个点")
    print(f"H3令牌化方法: {h3_method}")
    print(f"预估词表大小: {estimate_vocab_size(h3_method)}")
    
    total_segments = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        # 使用tqdm显示进度条
        for filename in tqdm(os.listdir(input_folder), desc="Processing files"):
            if filename.endswith(".csv"):
                path = os.path.join(input_folder, filename)
                try:
                    # 将轨迹文件分段处理（使用指定的H3方法）
                    segments = process_track_file_to_segments(path, segment_size, h3_method)
                    
                    # 为每个分段创建一个训练样本
                    for segment in segments:
                        data_record = {
                            "text": segment["text"],
                            "label": segment["label"],
                            "segment_length": segment["segment_length"],
                            "h3_method": segment["h3_method"],
                            "source_file": filename
                        }
                        
                        # 将JSON对象写入文件，并换行
                        f.write(json.dumps(data_record) + '\n')
                        total_segments += 1
                        
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
    
    print(f"数据集创建完成，已保存至: {output_file}")
    print(f"总共生成 {total_segments} 个轨迹分段")

if __name__ == "__main__":
    # 定义你的原始数据文件夹路径
    source_data_folder = 'data/nj_train'
    
    # 定义输出的数据集文件名
    output_dataset_file = 'train_dataset_nj.jsonl'
    
    # 选择H3令牌化方法（推荐使用'hierarchical_optimized'或'grid_based'）
    h3_method = 'char_split'  # 可根据需要修改
    
    # 执行数据集创建
    create_dataset(source_data_folder, output_dataset_file, segment_size=30, h3_method=h3_method)