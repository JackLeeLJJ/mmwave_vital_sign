import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib
matplotlib.use('TKAgg')
from mmwave_data_process import generate_synthetic_data,loadData,get_matched_filelists
from test9_unet import UNet
# # 文件夹路径
train_wake_radar_dir = "../data/mmwaveData80/trainData/down_mmwaveData/train_wake/"
train_tired_radar_dir="../data/mmwaveData80/trainData/down_mmwaveData/train_tired/"
train_ecg_dir = "../data/mmwaveData80/trainData/ecgData"

test_wake_radar_dir = "../data/mmwaveData80/testData/down_mmwaveData/test_wake/"
test_tired_radar_dir = "../data/mmwaveData80/testData/down_mmwaveData/test_tired"
test_ecg_dir = "../data/mmwaveData80/testData/ecgData"

# 生成对齐1024维度的数据集
data,labels=loadData(1024,test_wake_radar_dir,test_ecg_dir)
dataset = TensorDataset(torch.stack(data), torch.stack(labels))
test_loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载时需先创建模型实例再加载参数
loaded_model = UNet().to(device)
loaded_model.load_state_dict(torch.load('../model/unet_model.pth'))
loaded_model.eval()



# 修改后的可视化函数
def visualize_comparison():
    with torch.no_grad():
        test_samples, test_labels = next(iter(test_loader))
        reconstructions = loaded_model(test_samples[:2].to(device))

    plt.figure(figsize=(15, 6))
    for i in range(2):
        plt.subplot(2, 1, i + 1)
        plt.plot(test_labels[i].numpy(), label='Original', alpha=0.7)
        plt.plot(reconstructions[i].cpu().numpy(), label='Reconstructed', alpha=0.7)
        plt.legend()
    plt.tight_layout()
    plt.show()


visualize_comparison()