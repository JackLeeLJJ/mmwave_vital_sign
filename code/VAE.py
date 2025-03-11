import torch  # 导入 PyTorch 库
from torch.utils.data import DataLoader  # 导入 DataLoader，用于批量加载数据
from torchvision.datasets import MNIST  # 导入 MNIST 数据集
from torchvision.transforms import transforms  # 导入转换操作，如将图像转为Tensor
from torch import nn  # 导入PyTorch的神经网络模块
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import matplotlib.pyplot as plt  # 导入matplotlib，用于绘图
import matplotlib  # 导入matplotlib的主模块
matplotlib.use('TkAgg')  # 设置matplotlib使用TkAgg后端显示图形

# 定义变分自编码器（VAE）类
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, gaussian_dim):
        super().__init__()  # 调用父类构造函数，初始化VAE模型

        # 编码器部分
        self.fc1 = nn.Sequential(  # 定义编码器的全连接层
            nn.Linear(in_features=input_dim, out_features=hidden_dim),  # 输入层到隐藏层
            nn.Tanh(),  # 激活函数
            nn.Linear(in_features=hidden_dim, out_features=256),  # 隐藏层到256维
            nn.Tanh(),  # 激活函数
        )

        # μ（均值）和logσ²（对数方差）
        self.mu = nn.Linear(in_features=256, out_features=gaussian_dim)  # 输出潜在空间的均值,gaussian_dim是潜空间的维度，潜变量的每个维度都是一个概率分布，这个分布是
        # 通过神经网络学习出来的，每次采样时从每个维度都采样构成解码器的输入，每个维度包含了原始数据某个方面的数据
        self.log_sigma = nn.Linear(in_features=256, out_features=gaussian_dim)  # 输出潜在空间的对数方差

        # 解码器部分（重构）
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=gaussian_dim, out_features=256),  # 从潜在空间到256维
            nn.Tanh(),  # 激活函数
            nn.Linear(in_features=256, out_features=512),  # 到512维
            nn.Tanh(),  # 激活函数
            nn.Linear(in_features=512, out_features=input_dim),  # 到输入维度大小
            nn.Sigmoid()  # 使用Sigmoid将输出限制在0到1之间
        )

    def forward(self, x):
        # 前向传播
        h = self.fc1(x)  # 输入x通过编码器的全连接层得到h

        # 计算潜在变量的均值和对数方差
        mu = self.mu(h)  # 得到潜在空间的均值
        log_sigma = self.log_sigma(h)  # 得到潜在空间的对数方差

        # 使用重参数化技巧生成潜在变量的样本
        h_sample = self.reparameterization(mu, log_sigma)

        # 通过解码器重构数据
        reconstruction = self.fc2(h_sample)

        return reconstruction, mu, log_sigma  # 返回重构数据、均值和对数方差

    def reparameterization(self, mu, log_sigma):
        # 重参数化技巧，用于生成潜在变量
        sigma = torch.exp(log_sigma * 0.5)  # 计算标准差
        e = torch.randn_like(input=sigma, device=device)  # 从标准正态分布中生成噪声

        result = mu + e * sigma  # 生成潜在变量样本

        return result

    def predict(self, new_x):  # 用于生成新的数据
        reconstruction = self.fc2(new_x)  # 使用解码器重构数据
        return reconstruction

# 训练模型的函数
def train():
    transformer = transforms.Compose([  # 创建一个图像转换操作，转为Tensor
        transforms.ToTensor(),
    ])
    data = MNIST("../data", transform=transformer, download=True)  # 下载并加载MNIST数据集

    dataloader = DataLoader(data, batch_size=128, shuffle=True)  # 创建数据加载器，用于批量加载数据

    model = VAE(784, 512, 20).to(device)  # 初始化VAE模型，输入维度784，隐藏层维度512，潜在空间维度20

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 使用Adam优化器
    loss_fn = nn.MSELoss(reduction="sum")  # 使用均方误差损失函数
    epochs = 10  # 设置训练轮数

    # 训练循环
    for epoch in torch.arange(epochs):
        all_loss = 0
        dataloader_len = len(dataloader.dataset)  # 获取数据集的长度

        # 遍历训练数据
        for data in tqdm(dataloader, desc="第{}轮梯度下降".format(epoch)):
            sample, label = data  # 获取一个批次的输入数据和标签
            sample = sample.to(device)  # 将数据移动到GPU（如果有的话）
            sample = sample.reshape(-1, 784)  # 将28x28的图像重塑为一维向量（784维）
            result, mu, log_sigma = model(sample)  # 使用模型进行预测

            loss_likelihood = loss_fn(sample, result)  # 计算重构的损失

            # 计算KL散度损失
            loss_KL = torch.pow(mu, 2) + torch.exp(log_sigma) - log_sigma - 1

            # 总损失 = 重构损失 + KL散度损失
            loss = loss_likelihood + 0.5 * torch.sum(loss_KL)

            # 梯度归零并进行反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                all_loss += loss.item()  # 累加损失

        # 打印每轮的平均损失
        print("函数损失为：{}".format(all_loss / dataloader_len))

        # 保存模型
        torch.save(model, "../model/VAE.pth")

# 预测和绘制结果
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 检查是否有GPU
    # train()  # 训练模型

    # 载入训练好的模型进行预测
    model = torch.load("../model/VAE.pth", map_location="cpu", weights_only=False)  # 载入模型

    # 生成20个随机样本进行预测
    x = torch.randn(size=(20, 20))  # 生成20个20维的随机潜在变量
    result = model.predict(x).detach().numpy()  # 预测生成的图像
    result = result.reshape(-1, 28, 28)  # 重塑预测结果为28x28的图像大小

    # 绘制预测的图像
    for i in range(20):
        plt.subplot(4, 5, i + 1)  # 创建子图
        plt.imshow(result[i])  # 绘制图像
        plt.gray()  # 使用灰度色图
    plt.show()  # 显示图像


    # 假设 y 是潜变量，已经通过模型进行处理生成重构图片
    y = torch.randn(size=(20, 784))  # 生成20个随机潜变量
    recon,mu,logvar = model(y)  # 使用模型进行图像重构
    recon=recon.detach().numpy()
    recon = recon.reshape(-1, 28, 28)  # 重塑为28x28图像
    y = y.detach().numpy()  # 将潜变量转为numpy数组
    y = y.reshape(-1, 28, 28)  # 重塑为28x28图像


    for i in range(20):
        plt.subplot(4, 5, i + 1)  # 创建子图
        plt.imshow(y[i])  # 绘制图像
        plt.gray()  # 使用灰度色图
    plt.show()  # 显示图像

    for i in range(20):
        plt.subplot(4, 5, i + 1)  # 创建子图
        plt.imshow(recon[i])  # 绘制图像
        plt.gray()  # 使用灰度色图

    plt.show()  # 显示图像
