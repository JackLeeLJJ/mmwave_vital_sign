from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import math
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端避免阻塞

matplotlib.use('TKAgg')
from mmwave_data_process import generate_synthetic_data, loadData, get_matched_filelists

# # 文件夹路径
train_wake_radar_dir = "../data/mmwaveData80/trainData/down_mmwaveData/train_wake/"
train_tired_radar_dir = "../data/mmwaveData80/trainData/down_mmwaveData/train_tired/"
train_ecg_dir = "../data/mmwaveData80/trainData/ecgData"

test_wake_radar_dir = "../data/mmwaveData80/testData/down_mmwaveData/test_wake/"
test_tired_radar_dir = "../data/mmwaveData80/testData/down_mmwaveData/test_tired"
test_ecg_dir = "../data/mmwaveData80/testData/ecgData"

# 数据加载函数
def load_optimized_data(seq_length=256):
    """加载并下采样数据到256维"""
    raw_data, labels = loadData(1024, train_wake_radar_dir, train_ecg_dir)

    # 使用最大池化下采样
    downsampled_data = []
    for sample in raw_data:
        sample_tensor = torch.tensor(sample).unsqueeze(0).unsqueeze(0)  # [1,1,1024]
        pooled = F.max_pool1d(sample_tensor, kernel_size=4)  # [1,1,256]
        downsampled_data.append(pooled.squeeze())
    return torch.stack(downsampled_data), torch.stack(labels)


# 加载优化后的数据集
data, labels = load_optimized_data(256)
dataset = TensorDataset(data, labels)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)


class FastTransformerVAE(nn.Module):
    def __init__(self, input_dim=256, latent_dim=32, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = 32  # 减小模型维度

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=128,
                    nhead=nhead,
                    dim_feedforward=256,
                    batch_first=True
                ),
                num_layers=num_layers
            )
        )

        # 潜在空间
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=128,
                    nhead=nhead,
                    dim_feedforward=256,
                    batch_first=True
                ),
                num_layers=num_layers
            ),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastTransformerVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)
scaler = torch.cuda.amp.GradScaler()  # 混合精度训练


# 训练函数
def train_epoch():
    model.train()
    total_loss = 0
    for inputs, _ in tqdm(train_loader, desc="Training"):
        inputs = inputs.to(device)

        optimizer.zero_grad()

        # 混合精度训练
        with torch.cuda.amp.autocast():
            recon, mu, logvar = model(inputs)
            mse_loss = F.mse_loss(recon, inputs)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse_loss + 0.1 * kld

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    avg_loss = train_epoch()
    print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")


# 可视化保存
def save_results():
    model.eval()
    with torch.no_grad():
        test_samples, _ = next(iter(train_loader))
        recon, _, _ = model(test_samples[:2].to(device))

    plt.figure(figsize=(12, 6))
    for i in range(2):
        plt.subplot(2, 1, i + 1)
        plt.plot(test_samples[i].numpy(), label='Original')
        plt.plot(recon[i].cpu().numpy(), label='Reconstructed', alpha=0.7)
        plt.legend()
    plt.show()


save_results()