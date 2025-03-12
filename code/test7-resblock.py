import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib

matplotlib.use('TKAgg')
from mmwave_data_process import generate_synthetic_data, loadData, get_matched_filelists

# # 文件夹路径
train_wake_radar_dir = "../data/mmwaveData80/trainData/down_mmwaveData/train_wake/"
train_tired_radar_dir = "../data/mmwaveData80/trainData/down_mmwaveData/train_tired/"
train_ecg_dir = "../data/mmwaveData80/trainData/ecgData"

test_wake_radar_dir = "../data/mmwaveData80/testData/down_mmwaveData/test_wake/"
test_tired_radar_dir = "../data/mmwaveData80/testData/down_mmwaveData/test_tired"
test_ecg_dir = "../data/mmwaveData80/testData/ecgData"

# 生成对齐1024维度的数据集
data, labels = loadData(1024, train_wake_radar_dir, train_ecg_dir)

dataset = TensorDataset(torch.stack(data), torch.stack(labels))
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主分支
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 捷径分支（当维度不匹配时）
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 残差连接
        return self.act2(out)

class ResidualBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主分支（转置卷积）
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels,
                                     kernel_size=5, padding=2,
                                     stride=stride, output_padding=stride-1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                             kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 捷径分支
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels,
                                 kernel_size=1, stride=stride,
                                 output_padding=stride-1),
                nn.BatchNorm1d(out_channels)
            )

        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.act2(out)
class VED(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=10):
        super().__init__()

        # 编码器结构（带残差）
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=4, padding=2),  # 1024 → 256
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),

            # 添加残差块
            ResidualBlock(4, 8, stride=4),  # 256 → 64
            ResidualBlock(8, 16, stride=4),  # 64 → 16
            ResidualBlock(16, 32, stride=4),  # 16 → 4

            nn.Flatten(),
            nn.Linear(32 * (input_dim // (4 ** 4)), 256)  # 32*(1024/64)=32 * 16=512 → 256
        )

        # 潜在空间映射
        # self.fc_mu = nn.Linear(256, latent_dim)
        # self.fc_var = nn.Linear(256, latent_dim)

        # 解码器结构
        # 解码器结构（带残差）
        self.decoder = nn.Sequential(
            nn.Linear(256, 32 * (input_dim // (4 ** 4))),  # 256 → 32 * 4=128
            nn.Unflatten(1, (32, input_dim // (4 ** 4))),  # → [32,4]

            # 使用转置卷积残差块
            ResidualBlockTranspose(32, 16, stride=4),  # 4 → 16
            ResidualBlockTranspose(16, 8, stride=4),  # 16 → 64
            ResidualBlockTranspose(8, 4, stride=4),  # 64 → 256

            # 最终输出层
            nn.ConvTranspose1d(4, 1, kernel_size=4, stride=4, padding=0),  # 256 → 1024
            nn.Tanh()
        )


    def forward(self, x):
        # 编码过程
        # mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        # return self.decode(z), mu, logvar
        z = self.encode(x)

        return self.decode(z)

    def encode(self, x):
        x = x.unsqueeze(1)  # [B,1024] → [B,1,1024]
        encoded = self.encoder(x)
        return encoded

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        decoded = self.decoder(z)  # [B,1,1024]
        return decoded.squeeze(1)  # [B,1024]


# 3. 训练配置验证
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VED(input_dim=1024).to(device)
optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, 200, 0.8)
gamma = 0.2  # 调整KL系数


# 4. 损失函数（保持形状兼容）
def loss_function(recon_x, x):
    # 重构损失（MSE + MAE混合）
    mse_loss = F.mse_loss(recon_x, x)
    # mae_loss = F.l1_loss(recon_x, x)
    # recon_loss = 0.7 * mse_loss + 0.3 * mae_loss
    # kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # kl = kl.mean()  # 取批次平均
    # KL散度（防止过正则化）
    # kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # return recon_loss + gamma * kl_div
    return mse_loss


# 5. 增强型训练循环
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        reconstructions = model(inputs)

        # 多目标损失
        loss = loss_function(reconstructions, targets)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        # if batch_idx % 50 == 0:
        #     print(f'Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)}]'
        #           f'\tLoss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    print(f'====> Epoch {epoch} Average loss: {avg_loss:.4f}')


num_epochs = 500
for epoch in range(1, num_epochs + 1):
    train(epoch)


# 7. 可视化验证（对比原始和重建信号）
def visualize_comparison():
    with torch.no_grad():
        test_samples, test_labels = next(iter(train_loader))
        reconstructions= model(test_samples[:2].to(device))

    plt.figure(figsize=(15, 6))
    for i in range(2):
        plt.subplot(2, 1, i + 1)
        plt.plot(test_labels[i].numpy(), label='Original', alpha=0.7)
        plt.plot(reconstructions[i].cpu().numpy(), label='Reconstructed', alpha=0.7)
        plt.legend()
    plt.tight_layout()
    plt.show()


visualize_comparison()