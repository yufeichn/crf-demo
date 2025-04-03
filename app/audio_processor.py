import numpy as np
import torch
import soundfile as sf
import queue
import time

def preprocess_audio(audio_chunk):
    """预处理音频数据为模型输入格式"""
    # 确保是立体声
    if len(audio_chunk.shape) == 1:
        audio_chunk = np.column_stack((audio_chunk, audio_chunk))
    
    # 归一化
    if np.max(np.abs(audio_chunk)) > 0:
        audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
    
    # 转换为torch张量，形状为 [2, T]
    audio_tensor = torch.FloatTensor(audio_chunk).transpose(0, 1)
    return audio_tensor.unsqueeze(0)  # 添加批次维度 [1, 2, T]

def load_audio_file(file_path):
    """加载音频文件"""
    audio_data, sample_rate = sf.read(file_path)
    print(f"加载音频文件: {file_path}, 采样率: {sample_rate}")
    
    # 确保音频是立体声
    if len(audio_data.shape) == 1:
        audio_data = np.column_stack((audio_data, audio_data))
        print("单声道音频已转换为立体声")
    
    return audio_data, sample_rate

def process_audio(audio_data, sample_rate, hop_size, audio_queue, is_running):
    """处理音频数据并将其放入队列"""
    hop_samples = int(hop_size * sample_rate)
    total_samples = len(audio_data)
    overlap_factor = 0.1  # 10%的重叠，确保平滑过渡
    
    # 为了缓冲输出，一开始快速放入几个片段，然后恢复正常速度
    initial_fast_chunks = 10
    print(f"准备音频数据，加快前{initial_fast_chunks}个片段...")
    
    for i in range(0, total_samples, hop_samples):
        if not is_running():
            break
            
        end_idx = min(i + hop_samples, total_samples)
        chunk = audio_data[i:end_idx]
        
        # 如果最后一块数据长度不足，则填充
        if len(chunk) < hop_samples:
            chunk = np.pad(chunk, ((0, hop_samples - len(chunk)), (0, 0)), 'constant')
        
        audio_queue.put(chunk)
        
        # 前几个片段快速处理，后面恢复正常
        if i // hop_samples < initial_fast_chunks:
            time.sleep(0.01)  # 快速处理
        else:
            # 减去一些时间用于处理
            effective_sleep = max(0.01, hop_size - 0.1)  # 确保至少有10ms的处理时间
            time.sleep(effective_sleep)  # 实时性模拟：等待与音频时长相似的时间，但留出处理时间 