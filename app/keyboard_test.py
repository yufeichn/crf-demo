#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
键盘输入测试脚本，用于检测Mac电脑命令行环境是否能检测到键盘按键
"""

import time
import sys
from pynput import keyboard

print("键盘输入测试程序")
print("按下 'l' 或 'r' 键将显示按键状态")
print("按下 'ESC' 键退出程序")
print("请确保本程序在终端/命令行中运行，而不是在IDE内部运行")

# 创建键盘监听器
key_status = {"l": False, "r": False}

def on_press(key):
    """按键按下回调函数"""
    try:
        if key.char == 'l':
            key_status["l"] = True
            print("\r按下: L键 [激活]     ", end="", flush=True)
        elif key.char == 'r':
            key_status["r"] = True
            print("\r按下: R键 [激活]     ", end="", flush=True)
    except AttributeError:
        if key == keyboard.Key.esc:
            # 按下ESC键退出
            print("\n程序退出")
            return False

def on_release(key):
    """按键释放回调函数"""
    try:
        if key.char == 'l':
            key_status["l"] = False
            print("\r释放: L键 [未激活]   ", end="", flush=True)
        elif key.char == 'r':
            key_status["r"] = False
            print("\r释放: R键 [未激活]   ", end="", flush=True)
    except AttributeError:
        pass

# 创建监听器
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)

# 启动监听器
listener.start()

try:
    # 保持程序运行
    while listener.is_alive():
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n程序被中断")
finally:
    # 确保清理
    listener.stop()

print("测试完成") 