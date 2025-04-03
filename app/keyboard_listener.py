from pynput import keyboard

class KeyboardHandler:
    def __init__(self, socketio):
        """初始化键盘处理器"""
        self.current_channel = "original"  # 当前播放频道: "original", "left", "right"
        self.key_status = {"l": False, "r": False}  # 按键状态
        self.socketio = socketio
        self.listener = None
    
    def start(self):
        """启动键盘监听"""
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        return self.listener
    
    def stop(self):
        """停止键盘监听"""
        if self.listener:
            self.listener.stop()
    
    def on_press(self, key):
        """按键按下回调函数"""
        try:
            if key.char == 'l':
                self.current_channel = "left"
                self.key_status["l"] = True
                self.key_status["r"] = False
                self.socketio.emit('key_update', {'l': True, 'r': False})
            elif key.char == 'r':
                self.current_channel = "right"
                self.key_status["l"] = False
                self.key_status["r"] = True
                self.socketio.emit('key_update', {'l': False, 'r': True})
        except AttributeError:
            pass
    
    def on_release(self, key):
        """按键释放回调函数"""
        try:
            if key.char == 'l' or key.char == 'r':
                self.current_channel = "original"
                self.key_status["l"] = False
                self.key_status["r"] = False
                self.socketio.emit('key_update', {'l': False, 'r': False})
        except AttributeError:
            pass
    
    def get_current_channel(self):
        """获取当前播放频道"""
        return self.current_channel
    
    def get_key_status(self):
        """获取当前按键状态"""
        return self.key_status 