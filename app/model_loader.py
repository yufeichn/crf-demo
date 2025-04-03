import torch
import torch.nn as nn
import sys
import os

# 添加父目录到路径以导入模型定义
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import AudioUNet

def load_model(model_path):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioUNet().to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理DataParallel保存的权重（移除'module.'前缀）
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # 加载处理后的权重
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print(f"模型已加载，使用设备: {device}")
    return model, device 