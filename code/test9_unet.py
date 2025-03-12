import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib
matplotlib.use('TKAgg')
from mmwave_data_process import generate_synthetic_data,loadData,get_matched_filelists

# # 文件夹路径
train_wake_radar_dir = "../data/mmwaveData80/trainData/down_mmwaveData/train_wake/"
train_tired_radar_dir="../data/mmwaveData80/trainData/down_mmwaveData/train_tired/"
train_ecg_dir = "../data/mmwaveData80/trainData/ecgData"

test_wake_radar_dir = "../data/mmwaveData80/testData/down_mmwaveData/test_wake/"
test_tired_radar_dir = "../data/mmwaveData80/testData/down_mmwaveData/test_tired"
test_ecg_dir = "../data/mmwaveData80/testData/ecgData"


# 生成对齐1024维度的数据集
data,labels=loadData(1024,train_wake_radar_dir,train_ecg_dir)

dataset = TensorDataset(torch.stack(data), torch.stack(labels))
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)


class UNet(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()

        # 编码器（下采样）
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )  # 1024 -> 512

        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )  # 512 -> 256

        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 15, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )  # 256 -> 128

        # 中间层
        self.bottleneck = nn.Sequential(
            nn.Conv1d(256, 512, 15, padding=7),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, 2, stride=2)
        )  # 128 -> 256

        # 解码器（上采样）
        self.dec1 = nn.Sequential(
            nn.Conv1d(256 + 128, 128, 15, padding=7),  # 修正通道数为384（256+128）
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 2, stride=2)
        )  # 256 -> 512

        self.dec2 = nn.Sequential(
            nn.Conv1d(64 + 64, 64, 15, padding=7),  # 修正通道数为128（64+64）
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 2, stride=2)
        )  # 512 -> 1024

        # 最终输出层
        self.final = nn.Sequential(
            nn.Conv1d(32, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B,1024] -> [B,1,1024]

        # 编码路径
        enc1 = self.enc1(x)  # [B,64,512]
        enc2 = self.enc2(enc1)  # [B,128,256]
        enc3 = self.enc3(enc2)  # [B,256,128]

        # 中间层
        bottleneck = self.bottleneck(enc3)  # [B,256,256]

        # 解码路径（带skip connections）
        dec1 = self.dec1(torch.cat([bottleneck, enc2], dim=1))  # [B,256+128=384,256]
        dec2 = self.dec2(torch.cat([dec1, enc1], dim=1))  # [B,64+64=128,512]

        # 最终输出
        out = self.final(dec2).squeeze(1)  # [B,1024]
        return out

# 简化后的损失函数（仅需重构损失）
def loss_function(recon_x, x):
    return F.mse_loss(recon_x, x) + F.l1_loss(recon_x, x) * 0.3

# 修改后的训练循环
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        reconstructions = model(inputs)

        loss = loss_function(reconstructions, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch} Loss: {avg_loss:.4f}')


# 在所有训练代码外层添加保护块
if __name__ == '__main__':
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型和优化器
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 训练循环
    num_epochs = 70
    for epoch in range(1, num_epochs + 1):
        train(epoch)

    # 模型保存
    torch.save(model.state_dict(), '../model/unet_model.pth')
    print("模型参数已保存到 unet_model.pth")