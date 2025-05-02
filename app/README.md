# 音频分离实时演示系统

这是一个专为音频源分离设计的实时推理演示应用程序。该应用允许用户通过网页界面直观地展示分离效果，并能够通过键盘按键或脑电信号实时切换播放不同的分离结果，方便用户对比原始音频与分离后的效果差异。

## 功能特点

- **实时音频源分离**：基于深度学习模型进行实时音频分离
- **连续流式音频播放**：无缝切换不同音频源，保证聆听体验
- **响应式网页界面**：实时展示按键状态和播放状态
- **多种分离模式**：
  - 按住'L'键播放左声道分离结果
  - 按住'R'键播放右声道分离结果
  - 不按任何键时自动播放原始立体声音频
  - **注意力导向播放**：根据输入（键盘或脑电）实时切换播放左侧或右侧分离的音频。
  - **可选相对音量**：通过 `--relative-volume` 参数，可以控制非注意侧声道以相对较小的音量播放，而非完全静音。
  - **原始音频播放**：无注意力输入时播放原始立体声音频。
- **多种输入方式**：
  - 键盘输入模式：通过按键'L'和'R'控制注意力方向
  - 脑电输入模式：通过脑电信号自动控制注意力方向
- **可配置参数**：支持多种参数调整，以适应不同性能设备和应用场景

## 系统要求

- Python 3.7 或更高版本
- 操作系统：Windows/macOS/Linux
- 至少 4GB RAM (推荐 8GB 以上)
- 支持浏览器访问

## 依赖项

- **PyTorch** (>=1.7.0)：深度学习框架
- **Flask** (>=2.0.0)：Web 服务器框架
- **Flask-SocketIO** (>=5.0.0)：提供 WebSocket 支持
- **SoundFile** (>=0.10.0)：音频文件处理
- **SoundDevice** (>=0.4.0)：音频播放
- **pynput** (>=1.7.0)：键盘监听
- **NumPy** (>=1.19.0)：数值计算
- **scikit-learn** (==1.3.0)：用于脑电模型（仅在使用脑电输入模式时需要）
- **pyriemann** (==0.7)：用于脑电数据处理（仅在使用脑电输入模式时需要）

## 安装指南

1. 克隆仓库到本地：

```bash
git clone https://github.com/yufeichn/crf-demo.git
cd crf-demo
```

2. 安装所需依赖：

```bash
# 基础依赖
pip install torch flask flask-socketio soundfile sounddevice pynput numpy

# 如果需要使用脑电输入模式，还需安装以下依赖
pip install scikit-learn pyriemann
```

或者使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基础用法

启动应用程序：

```bash
python -m app.main --input_file <音频文件路径> --model_path <模型权重文件路径>
```

示例：

```bash
python -m app.main --input_file ../test_tse/stereo/sample_0010.wav  --model_path app/checkpoints/best_model.pth
```

### 输入模式选择

应用支持两种输入模式来控制音频注意力方向：

1. **键盘输入模式**（默认）：
   ```bash
   python -m app.main --input_file <音频文件路径> --model_path <模型权重路径>
   ```
   
2. **脑电输入模式**：
   ```bash
   python -m app.main --input_file <音频文件路径> --model_path <模型权重路径> --input_mode eeg
   ```
   
   > 注意：使用脑电输入模式需要准备好相应的数据文件。详细说明请参阅 [脑电输入模式使用说明](app/README_EEG.md)。

### 高级配置参数

应用程序支持多种可选参数来优化性能：

- `--window_size <大小>`: 滑动窗口大小(秒)，默认为3.0。较大的窗口可能提供更好的分离效果，但需要更多计算资源
- `--hop_size <大小>`: 窗口滑动步长(秒)，默认为0.2。较小的值可使音频转换更平滑，但增加计算负担
- `--port <端口号>`: Web服务器端口，默认为5000
- `--buffer_size <大小>`: 音频缓冲区大小（处理队列长度），默认为 5。此值显著影响**切换延迟**。较小的值（如 5-10）延迟较低，但可能在处理能力不足时导致丢帧。较大的值（如 20-30）可提高播放流畅度，但增加延迟和内存使用。
- `--volume_balance`: 启用音量均衡功能，使分离的声音与原始音频音量接近，解决切换时音量差异过大的问题
- `--momentum <数值>`: 音量均衡动量参数，默认为0.9，值越大音量变化越平滑（取值范围0-1之间）
- `--relative-volume <数值>`: 非注意声道的相对音量（0.0 到 1.0 之间），默认为 0.0。0.0 表示非注意侧完全静音（原始模式）。大于 0 时，非注意侧也会播放声音，音量为注意侧的指定比例。例如，0.3 表示非注意侧音量为注意侧的 30%。
- `--input_mode <模式>`: 选择输入模式，可选"keyboard"或"eeg"，默认为"keyboard"
- `--eeg_data_dir <目录>`: EEG数据目录，默认为"./eeg_utils"，仅在使用脑电输入模式时有效

完整示例：

```bash
python -m app.main --input_file samples/music.wav --model_path models/separation_model.pth --window_size 4.0 --hop_size 0.1 --buffer_size 30 --volume_balance --momentum 0.85
```

使用脑电输入的示例：

```bash
python -m app.main --input_file samples/music.wav --model_path models/separation_model.pth --input_mode eeg --eeg_data_dir ./custom_eeg_data
```

### 使用界面

1. 启动应用后，在浏览器中访问 `http://localhost:5000`（或您指定的端口）
2. 网页界面将显示当前播放状态和按键状态
3. 根据所选输入模式进行交互：
   - **键盘输入模式**：
     - 按住 `L` 键：播放左声道分离结果
     - 按住 `R` 键：播放右声道分离结果
     - 不按任何键：播放原始立体声音频
   - **脑电输入模式**：
     - 系统自动根据脑电信号预测用户注意力方向
     - 预测结果为左侧：播放左声道分离结果
     - 预测结果为右侧：播放右声道分离结果
4. 界面将实时更新，显示当前播放的音频类型

## macOS 特别说明

在 macOS 系统中，pynput 可能需要额外的权限才能正常工作：

1. 打开系统偏好设置 → 安全性与隐私 → 隐私 → 辅助功能
2. 确保您的终端应用（如 Terminal 或 iTerm）已添加并勾选
3. 如果仍有问题，可以运行测试脚本检查键盘监听功能：

```bash
python -m app.keyboard_test
```

如果测试成功，键盘按键时终端会显示对应信息。

## 排错指南

### 音频播放不连续或卡顿

- 增加 `--buffer_size` 参数值（如：30-50）
- 减小 `--hop_size` 参数值（如：0.1-0.05）
- 增大 `--window_size` 参数值（如：4.0-5.0）
- 检查计算机性能，确保CPU能够实时处理音频
- 尝试使用GPU进行推理（设置 `--device cuda`）

### 注意力切换延迟过高

- **减小缓冲区大小**：尝试减小 `--buffer_size` 参数值（例如，`--buffer_size 5`）。这是影响延迟最主要的因素。
- **检查系统性能**：确保 CPU/GPU 能够实时完成推理和音频处理。如果处理速度跟不上 `hop_size` 定义的速率，延迟会累积。
- **优化 `hop_size`**：较小的 `hop_size` 会增加计算量，可能导致延迟。在保证平滑度的前提下适当增大 `hop_size` 可能有帮助。
- **内部延迟设置**：代码内部已将音频输出延迟设置为 `'low'`，以减少播放器本身的缓冲。

### 音频源切换时音量差异大

- 使用 `--volume_balance` 参数启用音量均衡功能
- 调整 `--momentum` 参数（0.7-0.95之间）以平衡响应速度和平滑度：
  - 值越大（如0.95）：音量变化更平滑，但响应较慢
  - 值越小（如0.7）：响应更快，但可能有轻微波动
- 确保分离模型训练充分，输出质量较高

### 网页界面无法连接

- 确认端口号没有被其他应用占用
- 检查防火墙设置是否阻止了连接
- 尝试使用不同的浏览器
- 检查控制台日志中的错误信息

### 键盘监听无响应

- 确保已授予应用程序必要的系统权限
- 重启应用程序或终端
- 在 macOS 上运行键盘测试脚本确认权限设置正确

### 脑电输入模式问题

- 确保数据文件已正确放置在指定目录中
- 检查控制台输出查找潜在的错误信息
- 确认文件名大小写正确
- 尝试使用绝对路径指定数据目录
- 详细排错信息请参阅 [脑电输入模式使用说明](app/README_EEG.md)

## 项目结构

- `main.py`: 主程序入口，解析命令行参数并初始化各组件
- `model_loader.py`: 负责加载和初始化音频分离模型
- `audio_processor.py`: 处理音频流，包括读取、预处理和后处理
- `inference_processor.py`: 执行模型推理，将音频分离为不同声道
- `audio_player.py`: 管理音频播放，处理实时切换逻辑
- `keyboard_listener.py`: 监听键盘输入，捕获用户交互
- `model_input_handler.py`: 处理脑电模型的输入，模拟按键事件
- `calibration_processor.py`: 运行脑电校准模型，生成预测结果
- `web_server.py`: 提供Web服务器功能，包括Socket通信
- `templates/index.html`: 网页前端界面模板
- `static/`: 存放CSS、JavaScript等静态资源
- `eeg_utils/`: 包含脑电信号处理相关代码和模型

## 贡献指南

欢迎提交问题报告、功能请求或直接贡献代码：

1. Fork 项目仓库
2. 创建您的功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件

## 联系方式

如有任何问题或建议，请通过以下方式联系我们：

- 电子邮件：zhangyufeichn@gmail.com
- GitHub Issues：https://github.com/yufeichn/crf-demo/issues 