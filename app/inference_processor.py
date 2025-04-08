import numpy as np
import torch
import threading
import queue
import time
from app.audio_processor import preprocess_audio

def inference_thread(model, device, audio_queue, output_queue, sample_rate, window_size, hop_size, is_running, 
                    volume_balance=False, momentum=0.9):
    """推理线程：处理音频队列中的数据并进行推理"""
    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)
    
    # 初始化滑动窗口
    buffer = np.zeros((window_samples, 2))
    overlap_factor = 0.1  # 重叠因子
    
    # 上一个片段的末尾，用于过渡混合
    prev_output_end = None
    
    # 反初始化标志，确保模型加载后立即开始处理
    initialized = False
    
    # 添加性能监控变量
    last_print_time = time.time()
    frames_processed = 0
    frames_dropped = 0
    
    # 初始化音量均衡参数
    left_volume_ratio = 1.0
    right_volume_ratio = 1.0
    
    print(f"启动推理线程，窗口大小: {window_size}秒，步长: {hop_size}秒")
    if volume_balance:
        print(f"已启用音量均衡功能，动量参数: {momentum}")
    
    # 尝试预热模型
    print("预热模型...")
    dummy_input = torch.zeros((1, 2, window_samples)).to(device)
    with torch.no_grad():
        model(dummy_input)
    print("模型预热完成")
    
    while is_running():
        try:
            # 检查输出队列状态
            queue_size = output_queue.qsize()
            if queue_size > output_queue.maxsize * 0.8:  # 队列接近满时
                # 增加等待时间，让播放器有时间处理
                time.sleep(0.05)
                continue
                
            # 从队列获取音频块
            if not audio_queue.empty():
                audio_chunk = audio_queue.get(timeout=0.1)
                
                # 更新滑动窗口
                buffer = np.vstack((buffer[hop_samples:], audio_chunk))
                
                # 预处理
                input_tensor = preprocess_audio(buffer)
                input_tensor = input_tensor.to(device)
                
                # 推理
                with torch.no_grad():
                    output = model(input_tensor)
                
                # 后处理
                output_np = output.cpu().numpy().squeeze().transpose(1, 0)
                
                # 只保留hop_size对应的输出部分
                output_chunk = output_np[-hop_samples:]
                
                # 应用平滑过渡
                if prev_output_end is not None and initialized:
                    # 创建重叠区域
                    overlap_size = int(hop_samples * overlap_factor)
                    
                    # 创建线性权重用于混合
                    fade_in = np.linspace(0, 1, overlap_size).reshape(-1, 1)
                    fade_out = 1 - fade_in
                    
                    # 应用混合到重叠区域
                    output_chunk[:overlap_size] = (
                        output_chunk[:overlap_size] * fade_in + 
                        prev_output_end[-overlap_size:] * fade_out
                    )
                
                # 保存当前片段末尾用于下一次混合
                prev_output_end = output_chunk
                initialized = True
                
                # 分离左右声道作为两个独立音频源
                original_chunk = buffer[-hop_samples:]
                
                # 音量均衡处理
                if volume_balance:
                    # 计算原始音频左右声道的RMS音量
                    original_left_rms = np.sqrt(np.mean(np.square(original_chunk[:, 0]))) + 1e-10
                    original_right_rms = np.sqrt(np.mean(np.square(original_chunk[:, 1]))) + 1e-10
                    
                    # 计算分离音频的RMS音量
                    separated_left_rms = np.sqrt(np.mean(np.square(output_chunk[:, 0]))) + 1e-10
                    separated_right_rms = np.sqrt(np.mean(np.square(output_chunk[:, 1]))) + 1e-10
                    
                    # 计算音量比例
                    current_left_ratio = original_left_rms / separated_left_rms if separated_left_rms > 1e-6 else 1.0
                    current_right_ratio = original_right_rms / separated_right_rms if separated_right_rms > 1e-6 else 1.0
                    
                    # 使用动量更新音量比例
                    left_volume_ratio = momentum * left_volume_ratio + (1 - momentum) * current_left_ratio
                    right_volume_ratio = momentum * right_volume_ratio + (1 - momentum) * current_right_ratio
                    
                    # 应用音量均衡
                    output_chunk[:, 0] = output_chunk[:, 0] * left_volume_ratio
                    output_chunk[:, 1] = output_chunk[:, 1] * right_volume_ratio
                
                left_channel = np.column_stack((output_chunk[:, 0], np.zeros_like(output_chunk[:, 0])))
                right_channel = np.column_stack((np.zeros_like(output_chunk[:, 1]), output_chunk[:, 1]))
                
                # 避免输出队列阻塞
                try:
                    output_queue.put({
                        "original": original_chunk.copy(),
                        "left": left_channel.copy(),
                        "right": right_channel.copy()
                    }, timeout=0.1)
                    frames_processed += 1
                except queue.Full:
                    frames_dropped += 1
                    # 队列满了，增加等待时间
                    time.sleep(0.05)
                    
                # 每秒打印一次性能统计
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    stats = f"推理性能统计 - 处理帧数: {frames_processed}, 丢弃帧数: {frames_dropped}, " \
                           f"队列大小: {queue_size}/{output_queue.maxsize}"
                    
                    if volume_balance:
                        stats += f", 左声道音量比: {left_volume_ratio:.3f}, 右声道音量比: {right_volume_ratio:.3f}"
                    
                    print(stats)
                    frames_processed = 0
                    frames_dropped = 0
                    last_print_time = current_time
                    
            else:
                time.sleep(0.01)  # 避免空转
        except Exception as e:
            print(f"推理线程出错: {e}")
            time.sleep(0.01) 