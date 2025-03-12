import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.fft
import random
from pathlib import Path
from torch.utils.data import Dataset
from mmwave_data_process import generate_synthetic_data,loadData,get_matched_filelists
# 自定义数据处理器
class DataProcessor:
    @staticmethod
    def normalize(data, mode='input'):
        """数据标准化
        mode: 'input' 雷达信号 / 'output' ECG信号
        """
        if mode == 'input':
            # 雷达信号标准化（保持分布）
            return (data - np.mean(data)) / (np.std(data) + 1e-8)
        else:
            # ECG信号归一化到[-1,1]
            data_min, data_max = np.min(data), np.max(data)
            return 2 * (data - data_min) / (data_max - data_min + 1e-8) - 1

    @staticmethod
    def process_file_pair(radar_path, ecg_path, seq_length=256):
        """处理单个文件对"""
        radar = np.load(radar_path).astype(np.float32)
        ecg = np.load(ecg_path).astype(np.float32)

        # 标准化处理
        radar = DataProcessor.normalize(radar, mode='input')
        ecg = DataProcessor.normalize(ecg, mode='output')

        # 分割样本
        samples = []
        for i in range(len(radar) // seq_length):
            start = i * seq_length
            end = start + seq_length
            samples.append((
                torch.FloatTensor(radar[start:end]),
                torch.FloatTensor(ecg[start:end])
            ))
        return samples


# 增强数据集类
class EnhancedDataset(Dataset):
    def __init__(self, radar_dir, ecg_dir, seq_length=256, augment=True):
        self.radar_files, self.ecg_files = get_matched_filelists(radar_dir, ecg_dir)
        self.seq_length = seq_length
        self.augment = augment
        self.samples = []

        # 加载并处理数据
        for r_path, e_path in zip(self.radar_files, self.ecg_files):
            self.samples.extend(DataProcessor.process_file_pair(r_path, e_path, seq_length))

    def __len__(self):
        return len(self.samples) * 2  # 数据量翻倍

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            # 数据增强
            orig_idx = idx % len(self.samples)
            radar, ecg = self.samples[orig_idx]

            # 添加随机噪声
            radar = radar + torch.randn_like(radar) * 0.1
            ecg = ecg + torch.randn_like(ecg) * 0.1

            # 随机时间偏移
            shift = random.randint(-10, 10)
            return torch.roll(radar, shift), torch.roll(ecg, shift)
        else:
            return self.samples[idx]


# 增强的VAE模型
class EnhancedVED(nn.Module):
    def __init__(self, input_dim=256, latent_dim=128):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=4, padding=3),  # 256 →64
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.Conv1d(16, 32, 5, stride=4, padding=2),  # 64 →16
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 64, 5, stride=4, padding=2),  # 16 →4
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(64 * (input_dim // (4 ** 3)), 512)
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)

        # 解码器（支持大范围输出）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * (input_dim // (4 ** 3))),
            View((-1, 64, input_dim // (4 ** 3))),

            nn.ConvTranspose1d(64, 32, 5, stride=4, padding=2, output_padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(32, 16, 5, stride=4, padding=2, output_padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(16, 1, 5, stride=4, padding=2, output_padding=3),
            nn.Linear(input_dim, input_dim)  # 线性层调整输出范围
        )

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, x):
        x = x.unsqueeze(1)
        return self.fc_mu(self.encoder(x)), self.fc_var(self.encoder(x))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).squeeze(1)


class View(nn.Module):
    """维度重塑模块"""

    def __init__(self, shape):
        super().__init__
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


# 改进的混合损失函数
class HybridLoss(nn.Module):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, recon, target, mu, logvar):
        # 时域损失
        mse = F.mse_loss(recon, target)
        mae = F.l1_loss(recon, target)

        # 频域损失
        recon_fft = torch.fft.fft(recon)
        target_fft = torch.fft.fft(target)
        freq_loss = F.mse_loss(recon_fft.real, target_fft.real) + \
                    F.mse_loss(recon_fft.imag, target_fft.imag)

        # KL散度 (动态权重)
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        alpha = torch.sigmoid(torch.tensor([target.max().abs() / 600]))[0]

        return 0.5 * mse + 0.3 * mae + 0.2 * freq_loss + alpha * self.gamma * kl_div


# 训练配置
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_dataset = EnhancedDataset(
        radar_dir="../data/mmwaveData80/trainData/down_mmwaveData/train_wake/",
        ecg_dir="../data/mmwaveData80/trainData/ecgData/",
        seq_length=256
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 模型初始化
    model = EnhancedVED(input_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = HybridLoss(gamma=0.1)

    # 训练循环
    best_loss = float('inf')
    for epoch in range(100):
        model.train()
        total_loss = 0

        for batch_idx, (radar, ecg) in enumerate(train_loader):
            radar, ecg = radar.to(device), ecg.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(radar)
            loss = criterion(recon, ecg, mu, logvar)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    # 可视化验证
    visualize_results(model, device)


def visualize_results(model, device):
    test_dataset = EnhancedDataset(
        radar_dir="../data/mmwaveData80/testData/down_mmwaveData/test_wake/",
        ecg_dir="../data/mmwaveData80/testData/ecgData/",
        seq_length=256,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True)

    with torch.no_grad():
        radar, ecg = next(iter(test_loader))
        recon, _, _ = model(radar.to(device))

        radar = radar.cpu().numpy()
        ecg = ecg.cpu().numpy()
        recon = recon.cpu().numpy()

        plt.figure(figsize=(15, 9))
        for i in range(3):
            plt.subplot(3, 2, 2 * i + 1)
            plt.plot(ecg[i], label='Original')
            plt.plot(recon[i], label='Reconstructed', alpha=0.7)
            plt.title(f"Sample {i + 1} - Time Domain")
            plt.legend()

            plt.subplot(3, 2, 2 * i + 2)
            plt.plot(np.abs(np.fft.fft(ecg[i]))[:100], label='Original')
            plt.plot(np.abs(np.fft.fft(recon[i]))[:100], label='Reconstructed', alpha=0.7)
            plt.title(f"Sample {i + 1} - Frequency Domain")
            plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()