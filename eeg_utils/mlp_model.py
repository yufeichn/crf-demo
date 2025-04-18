import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import random # 导入 random 模块

# 0. 设置随机种子以确保可复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. 定义 MLP 模型 (添加 Dropout)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob) # 添加 Dropout
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob) # 添加 Dropout
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out) # 应用 Dropout
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out) # 应用 Dropout
        out = self.fc3(out)
        return out

# 2. 加载和预处理数据
def load_and_preprocess_data(train_data_path='X_train.npy', train_labels_path='Y_train.npy',
                             test_data_path='X_test.npy', test_labels_path='Y_test.npy',
                             val_size=0.2, batch_size=16):
    print("Loading data...")
    X_train_raw = np.load(train_data_path)
    Y_train_raw = np.load(train_labels_path)
    X_test_raw = np.load(test_data_path)
    Y_test_raw = np.load(test_labels_path)

    print(f"Original X_train shape: {X_train_raw.shape}") # [60, 18, 500]
    print(f"Original Y_train shape: {Y_train_raw.shape}") # [60,]
    print(f"Original X_test shape: {X_test_raw.shape}")   # [60, 18, 500]
    print(f"Original Y_test shape: {Y_test_raw.shape}")   # [60,]

    # 展平数据: [Batch_size, Num_EEG_Channels * Num_Samples]
    num_train_samples, num_channels, num_timepoints = X_train_raw.shape
    num_test_samples = X_test_raw.shape[0]
    input_size = num_channels * num_timepoints

    X_train_flat = X_train_raw.reshape(num_train_samples, -1)
    X_test_flat = X_test_raw.reshape(num_test_samples, -1)

    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat) # 使用训练集的 scaler 转换测试集

    # 划分训练集和验证集
    X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(
        X_train_scaled, Y_train_raw, test_size=val_size, random_state=42, stratify=Y_train_raw
    )

    # 转换为 PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train_split)
    Y_train_tensor = torch.LongTensor(Y_train_split) # CrossEntropyLoss 需要 LongTensor
    X_val_tensor = torch.FloatTensor(X_val_split)
    Y_val_tensor = torch.LongTensor(Y_val_split)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    Y_test_tensor = torch.LongTensor(Y_test_raw)

    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False) # 测试时 batch_size=1 模拟逐个样本

    return train_loader, val_loader, test_loader, input_size

# 3. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    print("Start Training...")
    model.to(device)
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train

        # 验证
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = correct_val / total_val

        # print(f'Epoch [{epoch+1}/{num_epochs}], '
        #       f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
        #       f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

        # 保存最佳模型
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_model_state = model.state_dict()
            # 可以选择在这里保存模型到文件
            # torch.save(model.state_dict(), 'best_mlp_model.pth')

    print('Finished Training')
    if best_model_state:
        model.load_state_dict(best_model_state) # 加载验证集上表现最好的模型
    return model

# 4. 测试函数 (模拟实时)
def test_model_realtime_simulation(model, test_loader, device='cpu'):
    print("Start Testing (Real-time Simulation)...")
    model.to(device)
    model.eval()
    running_total = 0
    running_correct = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader): # batch_size=1
            # time.sleep(0.1) # 模拟数据到达间隔
            inputs, labels = inputs.to(device), labels.to(device)
            y_true = labels.item()

            # 前向传播
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = predicted.item()

            print(f"Sample {i+1}: Predict Label = {y_pred}, True Label = {y_true}")
            all_predictions.append(y_pred)
            all_labels.append(y_true)

            # 更新实时准确率
            running_total += 1
            if y_true == y_pred:
                running_correct += 1
            running_acc = running_correct / running_total
            print(f"-> Running ACC: {running_acc:.4f} ({running_correct}/{running_total})")

    # 计算总体准确率
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    total_accuracy = np.mean(all_predictions == all_labels)
    print("==================================================")
    print(f"总体准确率评估: {total_accuracy:.4f} ({np.sum(all_predictions == all_labels)}/{len(all_labels)})")
    print("==================================================")


# 5. 主程序
if __name__ == "__main__":
    # 设置随机种子
    SEED = 42
    set_seed(SEED)

    # 超参数
    HIDDEN_SIZE1 = 128
    HIDDEN_SIZE2 = 64
    NUM_CLASSES = 2 # 左/右
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 70 # 可以适当增加 epochs 配合正则化
    BATCH_SIZE = 16
    VAL_SPLIT_SIZE = 0.25 # 稍微增加验证集比例
    DROPOUT_PROB = 0.5 # Dropout 概率
    WEIGHT_DECAY = 1e-4 # 权重衰减系数

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    train_loader, val_loader, test_loader, input_size = load_and_preprocess_data(batch_size=BATCH_SIZE, val_size=VAL_SPLIT_SIZE)

    # 创建模型
    model = MLP(input_size=input_size, hidden_size1=HIDDEN_SIZE1, hidden_size2=HIDDEN_SIZE2, num_classes=NUM_CLASSES, dropout_prob=DROPOUT_PROB)

    # 定义损失函数和优化器 (添加 weight_decay)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 训练模型
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=device)

    # 测试模型 (模拟实时)
    test_model_realtime_simulation(trained_model, test_loader, device=device) 