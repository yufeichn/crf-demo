import os
from flask import Flask, render_template
import flask_socketio

def create_web_server():
    """创建Flask Web服务器"""
    # 确保模板目录存在
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    os.makedirs(template_dir, exist_ok=True)
    
    # 创建Flask应用
    app = Flask(__name__, template_folder=template_dir)
    socketio = flask_socketio.SocketIO(app, cors_allowed_origins="*")
    
    @app.route('/')
    def index():
        """主页"""
        return render_template('index.html')
    
    @socketio.on('connect')
    def handle_connect():
        """WebSocket连接事件"""
        pass  # 连接处理会在主应用中添加
    
    return app, socketio 