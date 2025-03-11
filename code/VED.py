import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

matplotlib.use('TKAgg')
import torch.nn.functional as F  # 导入 torch.nn.functional


# 1. 模拟数据生成（假设数据）
def generate_synthetic_data(num_samples=1000, seq_length=200):
    """生成包含心跳和呼吸信号的合成数据"""
    # 心跳信号（模拟BVP波形）
    t = np.linspace(0, 10, seq_length)
    heart_rate = 1.2  # Hz
    heart_signal = 0.5 * np.sin(2 * np.pi * heart_rate * t) + 0.2 * np.sin(4 * np.pi * heart_rate * t)

    # 呼吸信号（低频）
    resp_rate = 0.25  # Hz
    resp_signal = 1.0 * np.sin(2 * np.pi * resp_rate * t)

    # 混合信号（非线性叠加）
    mixed_signal = 0.7 * resp_signal + 0.3 * heart_signal + 0.1 * np.random.randn(seq_length)

    # I/Q通道（添加相位差）
    i_signal = mixed_signal * np.cos(0.1 * np.pi * t)
    q_signal = mixed_signal * np.sin(0.1 * np.pi * t)

    return torch.FloatTensor(i_signal), torch.FloatTensor(q_signal), torch.FloatTensor(heart_signal)


# 生成数据集
num_samples = 5000  # 设置生成的样本数量为5000
data_I, data_Q, labels = [], [], []  # 初始化空列表用于存储生成的I信号（I/Q通道）、Q信号、标签数据

# 使用循环生成num_samples个样本
for _ in range(num_samples):
    # 调用generate_synthetic_data函数生成合成数据
    # i 是 I 通道的信号，q 是 Q 通道的信号，h 是标签数据（心跳信号）
    i, q, h = generate_synthetic_data()
    data_I.append(i)  # 将I通道的信号添加到data_I列表中
    data_Q.append(q)  # 将Q通道的信号添加到data_Q列表中
    labels.append(h)  # 将标签数据（心跳信号）添加到labels列表中

# 使用TensorDataset将数据打包成PyTorch的Dataset对象
# dataset包括了三个张量：I通道数据、Q通道数据和标签数据
dataset = TensorDataset(torch.stack(data_I), torch.stack(data_Q), torch.stack(labels))

# 使用DataLoader来加载数据，batch_size为32，表示每次加载32个样本
# shuffle=True表示每次加载时会对数据进行洗牌，确保训练的随机性
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)



# 2. VED模型定义
class VED(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=64):
        super().__init__()

        # I/Q编码器
        self.encoder_I = self._build_encoder(input_dim)
        self.encoder_Q = self._build_encoder(input_dim)

        # 潜在空间映射
        self.fc_mu_I = nn.Linear(256, latent_dim)
        self.fc_var_I = nn.Linear(256, latent_dim)
        self.fc_mu_Q = nn.Linear(256, latent_dim)
        self.fc_var_Q = nn.Linear(256, latent_dim)

        # 解码器
        self.decoder = self._build_decoder(latent_dim * 2, input_dim)

    def _build_encoder(self, input_dim):
        return nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=4, padding=2),  # 添加padding
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(4, 8, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 16, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),  # 最后一层卷积
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * (input_dim // (4 ** 5)),256)  # 假设最终要映射到256维  # 修正全连接层输入维度
        )

    def _build_decoder(self, latent_dim, input_dim):
        return nn.Sequential(
            nn.Linear(latent_dim, 64 * input_dim//(4**5)),
            nn.Unflatten(1, (64, input_dim//(4**5))),

            # 增加一层转置卷积（共5层，与 Encoder 严格对称）
            nn.ConvTranspose1d(64, 32, 5, 4, 2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(32, 16, 5, 4, 2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(16, 8, 5, 4, 2, output_padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(8, 4, 5, 4, 2, output_padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),

            # 新增第五层转置卷积
            nn.ConvTranspose1d(4, 1, 5, 4, 2, output_padding=1),  # 直接输出单通道
            nn.Sigmoid()
        )

    def encode(self, x_i, x_q):
        h_i = self.encoder_I(x_i.unsqueeze(1))
        h_q = self.encoder_Q(x_q.unsqueeze(1))
        mu_i, logvar_i = self.fc_mu_I(h_i), self.fc_var_I(h_i)
        mu_q, logvar_q = self.fc_mu_Q(h_q), self.fc_var_Q(h_q)
        return mu_i, logvar_i, mu_q, logvar_q

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_i, z_q):
        z = torch.cat([z_i, z_q], dim=1)
        decoded_output = self.decoder(z)
        return decoded_output.squeeze(1)

    def forward(self, x_i, x_q):
        mu_i, logvar_i, mu_q, logvar_q = self.encode(x_i, x_q)
        z_i = self.reparameterize(mu_i, logvar_i)
        z_q = self.reparameterize(mu_q, logvar_q)
        return self.decode(z_i, z_q), mu_i, logvar_i, mu_q, logvar_q


# 3. 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VED().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
gamma = 0.2  # KL正则化权重


# 4. 损失函数
def loss_function(recon_x, x, mu_i, logvar_i, mu_q, logvar_q):
    # 重构损失
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL散度
    kl_i = -0.5 * torch.sum(1 + logvar_i - mu_i.pow(2) - logvar_i.exp())
    kl_q = -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp())

    return recon_loss + gamma * (kl_i + kl_q)


# 5. 训练循环
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (i, q, target) in enumerate(train_loader):
        i, q, target = i.to(device), q.to(device), target.to(device)

        optimizer.zero_grad()
        recon_batch, mu_i, logvar_i, mu_q, logvar_q = model(i, q)
        loss = loss_function(recon_batch, target, mu_i, logvar_i, mu_q, logvar_q)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset):.4f}')


# 训练模型
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    train(epoch)


# 6. 可视化结果
def plot_results():
    with torch.no_grad():
        sample_i, sample_q, sample_h = next(iter(train_loader))
        recon, _, _, _, _ = model(sample_i.to(device), sample_q.to(device))

    plt.figure(figsize=(12, 6))
    plt.plot(sample_h[0].numpy(), label="Ground Truth")
    plt.plot(recon.cpu()[0].numpy(), label="Reconstructed")
    plt.title("Waveform Comparison")
    plt.legend()
    plt.show()


plot_results()
