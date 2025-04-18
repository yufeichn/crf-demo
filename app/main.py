import os
import sys
import argparse
import time
import threading
import queue

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model_loader import load_model
from app.audio_processor import load_audio_file, process_audio
from app.inference_processor import inference_thread
from app.audio_player import audio_playback_thread
from app.keyboard_listener import KeyboardHandler
from app.web_server import create_web_server
from app.model_input_handler import ModelInputHandler
from app.calibration_processor import CalibrationProcessor

def main():
    """主程序入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="音频分离实时推理")
    parser.add_argument("--input_file", type=str, required=True, help="输入音频文件路径 (.wav)")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重文件路径 (.pth)")
    parser.add_argument("--window_size", type=float, default=3.0, help="滑动窗口大小（秒）")
    parser.add_argument("--hop_size", type=float, default=0.2, help="窗口滑动步长（秒）")
    parser.add_argument("--port", type=int, default=5000, help="Web服务器端口")
    parser.add_argument("--buffer_size", type=int, default=20, help="音频缓冲区大小")
    parser.add_argument("--volume_balance", action="store_true", help="启用音量均衡，使分离的声音与原始音频音量接近")
    parser.add_argument("--momentum", type=float, default=0.9, help="音量均衡动量参数，越大音量变化越平滑（0-1之间）")
    parser.add_argument("--input_mode", type=str, choices=["keyboard", "eeg"], default="keyboard", 
                        help="选择输入模式：keyboard（键盘）或eeg（脑电）")
    parser.add_argument("--eeg_data_dir", type=str, default="./eeg_utils", 
                        help="EEG数据目录，包含X_train.npy、Y_train.npy、X_test.npy和Y_test.npy文件")
    
    args = parser.parse_args()
    
    # 全局变量
    audio_queue = queue.Queue(maxsize=args.buffer_size)
    output_queue = queue.Queue(maxsize=args.buffer_size)  # 增加队列大小，避免丢帧
    is_running = True  # 控制线程运行
    
    # 加载音频文件
    audio_data, sample_rate = load_audio_file(args.input_file)
    
    # 加载模型
    model, device = load_model(args.model_path)
    
    # 创建Web服务器
    app, socketio = create_web_server()
    
    # 定义当前使用的输入处理器
    input_handler = None
    calibration_processor = None
    
    # 根据输入模式选择不同的输入处理器
    if args.input_mode == "keyboard":
        print("使用键盘输入模式")
        # 创建键盘处理器
        input_handler = KeyboardHandler(socketio)
        # 启动键盘监听
        keyboard_listener = input_handler.start()
    else:  # eeg 模式
        print("使用脑电输入模式")
        # 创建模型输入处理器
        model_input_handler = ModelInputHandler(socketio)
        input_handler = model_input_handler  # 设置为主输入处理器
        # 启动模型输入处理
        model_input_listener = model_input_handler.start()
        
        # 创建并启动校准模型处理器
        calibration_processor = CalibrationProcessor(model_input_handler)
        calibration_processor.start(args.eeg_data_dir)
    
    # 定义is_running函数
    def get_is_running():
        return is_running
    
    # 定义获取当前频道的函数
    def get_current_channel():
        return input_handler.get_current_channel()
    
    # 添加WebSocket连接事件处理
    @socketio.on('connect')
    def handle_connect():
        """WebSocket连接事件"""
        key_status = input_handler.get_key_status()
        socketio.emit('key_update', {'l': key_status["l"], 'r': key_status["r"]})
    
    print(f"程序启动，正在初始化组件...")
    
    # 启动推理线程
    inference_thread_handle = threading.Thread(
        target=inference_thread, 
        args=(model, device, audio_queue, output_queue, sample_rate, args.window_size, 
              args.hop_size, get_is_running, args.volume_balance, args.momentum),
        name="InferenceThread"
    )
    inference_thread_handle.daemon = True
    inference_thread_handle.start()
    
    # 启动音频播放线程
    playback_thread = threading.Thread(
        target=audio_playback_thread,
        args=(output_queue, get_current_channel, sample_rate, get_is_running),
        name="PlaybackThread"
    )
    playback_thread.daemon = True
    playback_thread.start()
    
    # 给足够时间让音频回调启动
    time.sleep(1)
    
    # 启动音频处理主循环
    audio_thread = threading.Thread(
        target=process_audio,
        args=(audio_data, sample_rate, args.hop_size, audio_queue, get_is_running),
        name="AudioProcessThread"
    )
    audio_thread.daemon = True
    audio_thread.start()
    
    # 启动Web服务器
    try:
        print(f"服务器启动，在浏览器中访问 http://localhost:{args.port}")
        socketio.run(app, host='0.0.0.0', port=args.port, debug=False)
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        is_running = False
        input_handler.stop()
        if calibration_processor:
            calibration_processor.stop()
        print("程序已停止")

if __name__ == "__main__":
    main() 