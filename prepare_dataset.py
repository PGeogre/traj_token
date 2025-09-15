# /home/maomao/project/token/prepare_dataset.py
import os
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime
import h3
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


def h3_to_tokens_cell_based(h3_str):
    """
    将H3字符串转换为基于单元格的令牌序列
    输出格式: H3_res_几, H3_base_几, H3_cell_1_几, H3_cell_2_几, H3_cell_3_几, ...
    
    参数:
        h3_str: H3索引字符串
    
    返回:
        tokens: 令牌列表，包含分辨率、基础单元格和各级子单元格
    """
    try:
        # 验证H3字符串有效性
        if not h3.is_valid_cell(h3_str):
            return [f"H3_INVALID_{h3_str}"]

        # 获取H3的分辨率和基础单元格
        res = h3.get_resolution(h3_str)
        base_cell = h3.get_base_cell_number(h3_str)
        h3_int = h3.str_to_int(h3_str)

        tokens = [
            f"H3_res_{res}",        # 分辨率令牌 (0-15)
            f"H3_base_{base_cell}" # 基础单元格令牌 (122个基础单元格)
        ]

        # 提取各级子单元格的方向信息
        # H3使用3位编码表示7个方向 (0-6，7是无效值)
        for level in range(res):
            shift = 45 - 3 * level  # H3内部使用位移来编码层级信息
            if shift < 0:  # 防止超出范围
                break
            direction = (h3_int >> shift) & 0b111  # 提取3位方向信息
            tokens.append(f"H3_cell_{level + 1}_{direction}")

        return tokens
        
    except Exception as e:
        # 处理任何H3库可能抛出的异常
        return [f"H3_ERROR_{str(e).replace(' ', '_')}"]



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
    'cell_based': h3_to_tokens_cell_based,  # 基于单元格的方案
}

def get_h3_tokenizer(method='cell_based'):
    """获取指定的H3令牌化函数"""
    return H3_TOKENIZATION_OPTIONS.get(method, h3_to_tokens_cell_based)

def estimate_vocab_size(method='cell_based'):
    """估算H3令牌化方案的词表大小"""
    if method == 'cell_based':
        return {
            'H3_res_*': 16,         # 分辨率0-15
            'H3_base_*': 122,       # H3有122个基础单元格
            'H3_cell_*_*': 7 * 15,  # 每个分辨率级别最多7个方向，最多15级
            'H3_INVALID_*': 'variable',
            'total_estimate': '< 300 tokens'
        }
    return '未知'



# ====================================================================
# 2. 核心处理函数，将单个CSV文件中的每个轨迹点转换为令牌序列
# ====================================================================
def process_track_file_to_segments(csv_path, segment_size=30, h3_method='cell_based'):
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
        
        for _, row in segment_df.iterrows():
            # 为每个轨迹点生成令牌序列
            point_tokens = []
            
            # 1. 时间令牌（拆分为具体组件）
            time_tokens = time_to_tokens(row['date'])
            point_tokens.extend(time_tokens)
            
            # 2. H3位置令牌
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
def create_dataset(input_folder, output_file, segment_size=30, h3_method='cell_based'):
    """
    处理文件夹中的所有CSV文件，将它们按分段处理并保存为JSONL格式。
    每个分段包含指定数量的轨迹点和对应的类别标签。
    
    参数:
    - h3_method: H3令牌化方案，当前使用基于单元格的方案 'cell_based'
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
    source_data_folder = 'data/data_demo'
    
    # 定义输出的数据集文件名
    output_dataset_file = 'train_dataset_demo.jsonl'
    
    # 使用基于单元格的H3令牌化方法
    h3_method = 'cell_based'
    
    # 执行数据集创建
    create_dataset(source_data_folder, output_dataset_file, segment_size=30, h3_method=h3_method)