import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class AudioUNet(nn.Module):
    def __init__(self):
        super(AudioUNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(2, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        
        # Decoder
        self.dec4 = ConvBlock(256 + 128, 128)
        self.dec3 = ConvBlock(128 + 64, 64)
        self.dec2 = ConvBlock(64 + 32, 32)
        self.dec1 = ConvBlock(32 + 2, 2)
        
        # Final output
        self.final = nn.Conv1d(2, 2, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder path with skip connections
        dec4 = self.dec4(torch.cat([self.upsample(enc4), enc3], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec4), enc2], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc1], dim=1))
        dec1 = self.dec1(torch.cat([dec2, x], dim=1))
        
        # Final output
        out = self.final(dec1)
        return out 

if __name__ == "__main__":
    import time

    # 设定测试参数
    device = torch.device("cpu") # 或者 torch.device("cuda") 如果有GPU
    # 模拟一个输入片段，例如3秒，16000Hz采样率
    # batch_size=1, channels=2, sequence_length=3*16000
    dummy_input = torch.randn(1, 2, 3 * 16000).to(device)
    num_runs = 100 # 测试运行次数

    # 实例化模型
    model = AudioUNet().to(device)
    model.eval() # 设置为评估模式

    # 1. 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数数量: {total_params:,}")

    # 2. 测试推理延迟
    # 热身运行
    with torch.no_grad():
        _ = model(dummy_input)

    # 多次运行计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_runs) * 1000

    print(f"设备: {device}")
    print(f"输入尺寸: {list(dummy_input.shape)}")
    print(f"运行次数: {num_runs}")
    print(f"平均推理延迟: {avg_latency_ms:.3f} ms") 