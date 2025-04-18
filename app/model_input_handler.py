import threading
import time
import sys
import os
import numpy as np
from queue import Queue


class ModelInputHandler:
    def __init__(self, socketio):
        """初始化模型输入处理器"""
        self.current_channel = "original"  # 当前播放频道: "original", "left", "right"
        self.key_status = {"l": False, "r": False}  # 按键状态
        self.socketio = socketio
        self.thread = None
        self.is_running = False
        self.input_queue = Queue()
        
    def start(self):
        """启动模型输入监听线程"""
        self.is_running = True
        self.thread = threading.Thread(target=self.process_model_input, daemon=True)
        self.thread.start()
        return self.thread
    
    def stop(self):
        """停止模型输入监听"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def process_model_input(self):
        """处理来自模型的输入"""
        while self.is_running:
            if not self.input_queue.empty():
                prediction = self.input_queue.get()
                self.update_channel(prediction)
            time.sleep(0.1)  # 防止CPU使用率过高
    
    def update_channel(self, prediction):
        """根据模型预测更新通道"""
        if prediction == 0:  # 左侧
            self.current_channel = "left"
            self.key_status["l"] = True
            self.key_status["r"] = False
            self.socketio.emit('key_update', {'l': True, 'r': False})
        elif prediction == 1:  # 右侧
            self.current_channel = "right"
            self.key_status["l"] = False
            self.key_status["r"] = True
            self.socketio.emit('key_update', {'l': False, 'r': True})
        # 释放按键状态的操作将由外部控制，通常是下一个预测到来或超时
    
    def reset_channel(self):
        """重置通道状态"""
        self.current_channel = "original"
        self.key_status["l"] = False
        self.key_status["r"] = False
        self.socketio.emit('key_update', {'l': False, 'r': False})
    
    def add_prediction(self, prediction):
        """添加模型预测结果到队列"""
        self.input_queue.put(prediction)
    
    def get_current_channel(self):
        """获取当前播放频道"""
        return self.current_channel
    
    def get_key_status(self):
        """获取当前按键状态"""
        return self.key_status 