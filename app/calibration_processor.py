import threading
import time
import sys
import os
import numpy as np


class CalibrationProcessor:
    def __init__(self, model_input_handler):
        """初始化校准模型处理器"""
        self.model_input_handler = model_input_handler
        self.thread = None
        self.is_running = False
        self.whitening_matrix = None
        self.model_weights = None
        
    def start(self, eeg_data_dir):
        """启动校准模型处理器"""
        self.is_running = True
        self.thread = threading.Thread(
            target=self._run_calibration_model, 
            args=(eeg_data_dir,),
            daemon=True
        )
        self.thread.start()
        return self.thread
    
    def stop(self):
        """停止校准模型处理器"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _run_calibration_model(self, eeg_data_dir):
        """运行校准模型"""
        try:
            # 添加EEG工具路径到系统路径
            eeg_utils_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if eeg_utils_path not in sys.path:
                sys.path.append(eeg_utils_path)
                print(f"添加路径到sys.path: {eeg_utils_path}")
            
            # 打印当前工作目录
            print(f"当前工作目录: {os.getcwd()}")
            
            # 尝试导入校准模型
            try:
                from eeg_utils.calibration_model import (
                    whitening_transformation_train_reg, 
                    train_svm, 
                    whitening_transformation_test,
                    test_single
                )
                print("成功导入calibration_model模块")
            except ImportError as e:
                print(f"导入calibration_model模块失败: {e}")
                return
            
            # 检查eeg_data_dir是否存在
            if not os.path.exists(eeg_data_dir):
                print(f"错误: 目录 {eeg_data_dir} 不存在!")
                return
            
            print(f"使用EEG数据目录: {eeg_data_dir}")
            
            # 构建数据文件路径
            X_train_path = os.path.join(eeg_data_dir, 'X_train.npy')
            Y_train_path = os.path.join(eeg_data_dir, 'Y_train.npy')
            X_test_path = os.path.join(eeg_data_dir, 'X_test.npy')
            Y_test_path = os.path.join(eeg_data_dir, 'Y_test.npy')
            
            # 检查文件是否存在
            print("检查数据文件:")
            for path, name in zip(
                [X_train_path, Y_train_path, X_test_path, Y_test_path],
                ['X_train.npy', 'Y_train.npy', 'X_test.npy', 'Y_test.npy']
            ):
                if os.path.exists(path):
                    print(f"√ 找到文件: {name}")
                else:
                    print(f"× 找不到文件: {name}")
            
            # 尝试在calibration_model.py所在的目录中查找数据文件
            if not all(os.path.exists(p) for p in [X_train_path, Y_train_path, X_test_path, Y_test_path]):
                print("在指定目录中找不到所有数据文件，正在尝试在calibration_model.py所在目录中查找...")
                
                # 查找calibration_model.py所在目录
                import eeg_utils
                eeg_utils_dir = os.path.dirname(eeg_utils.__file__)
                print(f"eeg_utils目录: {eeg_utils_dir}")
                
                # 重新构建数据文件路径
                X_train_path = os.path.join(eeg_utils_dir, 'X_train.npy')
                Y_train_path = os.path.join(eeg_utils_dir, 'Y_train.npy')
                X_test_path = os.path.join(eeg_utils_dir, 'X_test.npy')
                Y_test_path = os.path.join(eeg_utils_dir, 'Y_test.npy')
                
                # 再次检查文件
                print("在eeg_utils目录中检查数据文件:")
                for path, name in zip(
                    [X_train_path, Y_train_path, X_test_path, Y_test_path],
                    ['X_train.npy', 'Y_train.npy', 'X_test.npy', 'Y_test.npy']
                ):
                    if os.path.exists(path):
                        print(f"√ 找到文件: {name}")
                    else:
                        print(f"× 找不到文件: {name}")
            
            if not all(os.path.exists(p) for p in [X_train_path, Y_train_path, X_test_path, Y_test_path]):
                print("无法找到所有必需的EEG数据文件，请确保已放置在正确位置")
                return
            
            # 尝试加载数据文件
            try:
                print("正在加载数据文件...")
                X_train = np.load(X_train_path)
                print(f"成功加载 X_train，形状: {X_train.shape}")
                
                Y_train = np.load(Y_train_path)
                print(f"成功加载 Y_train，形状: {Y_train.shape}")
                
                X_test = np.load(X_test_path)
                print(f"成功加载 X_test，形状: {X_test.shape}")
                
                Y_test = np.load(Y_test_path)
                print(f"成功加载 Y_test，形状: {Y_test.shape}")
            except Exception as e:
                print(f"加载数据文件时出错: {e}")
                import traceback
                traceback.print_exc()
                return
            
            print("开始训练校准模型...")
            # 训练模型
            try:
                X_train_whitened, self.whitening_matrix = whitening_transformation_train_reg(X_train, reg=1e-8)
                self.model_weights = train_svm(X_train_whitened, Y_train)
                print("校准模型训练完成")
            except Exception as e:
                print(f"训练模型时出错: {e}")
                import traceback
                traceback.print_exc()
                return
            
            print("开始实时预测...")
            test_num = len(Y_test)
            
            # 实时模拟预测
            for i in range(0, test_num):
                if not self.is_running:
                    print("检测到停止信号，终止预测")
                    break
                    
                # 处理当前数据
                x_test = X_test[i, :, :]
                y_test = Y_test[i]
                
                try:
                    # 数据校准和预测
                    x_test_whitened = whitening_transformation_test(x_test, self.whitening_matrix)
                    y_pred = test_single(x_test_whitened, y_test, self.model_weights)
                    
                    # 将预测结果发送到输入处理器
                    self.model_input_handler.add_prediction(y_pred)
                    
                    # 暂停1秒，模拟实时数据输入
                    time.sleep(1)
                    
                    # 每次预测后重置通道
                    self.model_input_handler.reset_channel()
                except Exception as e:
                    print(f"预测过程中出错: {e}")
                    import traceback
                    traceback.print_exc()
                    # 继续处理下一帧
            
            print("校准模型预测完成")
            
        except Exception as e:
            print(f"校准模型运行异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            print("校准模型处理器已停止") 