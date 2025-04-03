import time
import numpy as np
import sounddevice as sd
import threading
import collections

def audio_playback_thread(output_queue, current_channel, sample_rate, is_running):
    """音频播放线程：从输出队列获取分离后的音频并根据当前选择的通道播放"""
    # 创建缓冲区
    buffer_size = 10  # 增加缓冲区大小
    audio_buffer = collections.deque(maxlen=buffer_size)
    
    # 性能监控变量
    last_print_time = time.time()
    frames_played = 0
    buffer_underruns = 0
    
    # 创建回调函数用于流式播放
    def audio_callback(outdata, frames, time, status):
        nonlocal frames_played, buffer_underruns
        
        if status:
            print(f"音频回调状态: {status}")
        
        if len(audio_buffer) == 0:
            # 缓冲区为空，填充静音
            outdata[:] = np.zeros((frames, 2))
            buffer_underruns += 1
            return
            
        # 从缓冲区获取音频
        current_chunk = audio_buffer.popleft()
        frames_played += frames
        
        # 如果音频片段小于所需帧数，填充静音
        if len(current_chunk) < frames:
            # 填充零
            padded = np.zeros((frames, 2))
            padded[:len(current_chunk)] = current_chunk
            outdata[:] = padded
        else:
            # 使用音频片段
            outdata[:] = current_chunk[:frames]
            
            # 如果有剩余，放回缓冲区前端
            if len(current_chunk) > frames:
                audio_buffer.appendleft(current_chunk[frames:])

    # 初始预填充缓冲区
    print("预填充音频缓冲区...")
    buffer_fill_count = 0
    last_channel = None
    
    # 打开音频流
    with sd.OutputStream(samplerate=sample_rate, channels=2, 
                         callback=audio_callback, blocksize=int(0.1 * sample_rate)):
        while is_running():
            try:
                current_ch = current_channel()
                
                # 如果缓冲区不够满或者通道改变，尝试填充
                if len(audio_buffer) < buffer_size or current_ch != last_channel:
                    # 获取新音频
                    if not output_queue.empty():
                        audio_dict = output_queue.get(timeout=0.1)
                        
                        # 根据当前选择的通道选择音频
                        audio_to_play = audio_dict[current_ch]
                        audio_buffer.append(audio_to_play)
                        
                        last_channel = current_ch
                        
                        if buffer_fill_count < buffer_size:
                            buffer_fill_count += 1
                            if buffer_fill_count == buffer_size:
                                print("缓冲区已填满，开始播放")
                    else:
                        time.sleep(0.01)  # 短暂休眠，避免空转
                else:
                    time.sleep(0.01)  # 短暂休眠，避免空转
                
                # 每秒打印一次性能统计
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    print(f"播放性能统计 - 播放帧数: {frames_played}, 缓冲区下溢: {buffer_underruns}, "
                          f"缓冲区大小: {len(audio_buffer)}/{buffer_size}")
                    frames_played = 0
                    buffer_underruns = 0
                    last_print_time = current_time
                        
            except Exception as e:
                print(f"音频播放线程出错: {e}")
                time.sleep(0.1) 