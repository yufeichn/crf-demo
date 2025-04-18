# 脑电输入模式使用说明

本应用支持两种输入模式来控制音频注意力方向：
1. 键盘输入模式（默认）：通过按下 'L' 和 'R' 键来控制注意力方向
2. 脑电输入模式：通过脑电 (EEG) 校准模型的实时预测结果来控制注意力方向

## 准备工作

使用脑电输入模式前，需确保数据文件已准备就绪：

1. 默认情况下，程序会在 `eeg_utils` 目录中查找脑电数据文件
2. 需要确保以下四个 NumPy 数据文件存在于指定目录中：
   - `X_train.npy` - 训练数据输入
   - `Y_train.npy` - 训练数据标签
   - `X_test.npy` - 测试数据输入
   - `Y_test.npy` - 测试数据标签

这些文件的格式应与 `eeg_utils/calibration_model.py` 中使用的格式一致：
- X_train 和 X_test 的形状为 [Batch_size, Num_EEG_Channels, Num_Samples]
- Y_train 和 Y_test 的形状为 [Batch_size]，其中 0 表示左侧注意力，1 表示右侧注意力

## 数据文件路径问题

为避免路径问题，请注意以下几点：

1. 确保数据文件放在正确的位置：
   - 相对路径：程序运行目录下的 `eeg_utils` 目录中
   - 也可通过 `--eeg_data_dir` 参数指定绝对路径

2. 文件名需要完全匹配（区分大小写）：
   - `X_train.npy`（不是 `x_train.npy` 或 `X_Train.npy`）
   - `Y_train.npy`（不是 `y_train.npy` 或 `Y_Train.npy`）
   - `X_test.npy`（不是 `x_test.npy` 或 `X_Test.npy`）
   - `Y_test.npy`（不是 `y_test.npy` 或 `Y_Test.npy`）

## 启动应用

### 键盘输入模式（默认）

```bash
python -m app.main --input_file <音频文件路径> --model_path <模型权重路径> [其他参数]
```

### 脑电输入模式

```bash
python -m app.main --input_file <音频文件路径> --model_path <模型权重路径> --input_mode eeg [其他参数]
```

默认情况下，程序会在 `eeg_utils` 目录中查找数据文件。如需指定其他目录：

```bash
python -m app.main --input_file <音频文件路径> --model_path <模型权重路径> --input_mode eeg --eeg_data_dir <EEG数据目录> [其他参数]
```

### 完整参数列表

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --input_file | 输入音频文件路径 (.wav) | 必填 |
| --model_path | 模型权重文件路径 (.pth) | 必填 |
| --window_size | 滑动窗口大小（秒） | 3.0 |
| --hop_size | 窗口滑动步长（秒） | 0.2 |
| --port | Web服务器端口 | 5000 |
| --buffer_size | 音频缓冲区大小 | 20 |
| --volume_balance | 启用音量均衡 | False |
| --momentum | 音量均衡动量参数 | 0.9 |
| --input_mode | 输入模式：keyboard（键盘）或eeg（脑电） | keyboard |
| --eeg_data_dir | EEG数据目录 | ./eeg_utils |

## 工作原理

在脑电输入模式下，应用会：

1. 加载并训练脑电校准模型（使用 X_train 和 Y_train 数据）
2. 对测试数据（X_test）进行一系列实时预测
3. 每秒生成一个预测结果（0 表示左侧注意力，1 表示右侧注意力）
4. 将预测结果转换为对应的注意力方向控制：
   - 0 → 左侧注意力（相当于按下 'L' 键）
   - 1 → 右侧注意力（相当于按下 'R' 键）

当每次新的预测结果到来时，前一个注意力状态会被重置。

## 故障排除

### 文件未找到错误

如果看到 `[Errno 2] No such file or directory: 'X_train.npy'` 或类似错误：

1. 检查文件路径是否正确
   ```bash
   # 在程序根目录下运行
   ls -la eeg_utils/
   # 确认文件存在且命名正确
   ```

2. 尝试直接指定绝对路径
   ```bash
   python -m app.main --input_file <音频文件路径> --model_path <模型权重路径> --input_mode eeg --eeg_data_dir /home/用户名/完整路径/eeg_utils
   ```

3. 确认数据文件没有被重命名或损坏

### 模型训练或预测异常

1. 检查数据格式是否符合要求
   - X_train 和 X_test 必须是三维数组 [Batch_size, Num_EEG_Channels, Num_Samples]
   - Y_train 和 Y_test 必须是一维数组 [Batch_size]，且值为0或1

2. 确认 NumPy 版本兼容性
   ```bash
   pip show numpy
   # 确保版本为1.19.0或更高
   ```

3. 如需调试，可修改 `calibration_processor.py` 添加更多日志输出

### 实时预测无响应

1. 确保应用程序具有足够的系统资源
2. 检查终端输出，查找潜在的错误信息或警告
3. 确认脑电模型没有异常中断

## 自定义开发

如需修改或扩展脑电输入功能，可以编辑以下文件：

- `app/model_input_handler.py`: 控制如何处理脑电模型的预测结果
- `app/calibration_processor.py`: 包含脑电校准模型的加载和预测逻辑
- `eeg_utils/calibration_model.py`: 实现校准模型的核心算法 