import os
import numpy as np
import re
from natsort import natsorted
import torch


# # 文件夹路径
train_wake_radar_dir = "../data/mmwaveData80/trainData/down_mmwaveData/train_wake/"
train_tired_radar_dir="../data/mmwaveData80/trainData/down_mmwaveData/train_tired/"
train_ecg_dir = "../data/mmwaveData80/trainData/ecgData"

test_wake_radar_dir = "../data/mmwaveData80/testData/down_mmwaveData/test_wake/"
test_tired_radar_dir = "../data/mmwaveData80/testData/down_mmwaveData/test_tired"
test_ecg_dir = "../data/mmwaveData80/testData/ecgData"

#

def get_matched_filelists(radar_dir, ecg_dir):
    """改进后的安全匹配函数"""
    # 构建ECG文件索引
    ecg_index = {}
    for f in os.listdir(ecg_dir):
        if f.startswith("ECG_Clean") and f.endswith(".npy"):
            # 使用更精确的正则表达式
            match = re.match(r"ECG_Clean_(\d{4})_([a-z]+)_(\d+)_(\d+)\.npy", f)
            if match:
                date, exp, subj, trial = match.groups()
                key = f"{date}_{exp}_{subj}_{trial}"
                ecg_index[key] = os.path.join(ecg_dir, f)

    # 匹配雷达文件
    matched_pairs = []
    for radar_file in natsorted(os.listdir(radar_dir)):
        if radar_file.startswith("mmwave") and radar_file.endswith(".npy"):
            # 解析雷达文件名
            radar_match = re.match(r"mmwave_(\d{4})_([a-z]+)_(\d+)_(\d+)_(\d+)\.npy", radar_file)
            if radar_match:
                r_date, r_exp, r_subj, r_trial, _ = radar_match.groups()
                ecg_key = f"{r_date}_{r_exp}_{r_subj}_{r_trial}"

                if ecg_key in ecg_index:
                    radar_path = os.path.join(radar_dir, radar_file)
                    matched_pairs.append((radar_path, ecg_index[ecg_key]))

    return list(zip(*matched_pairs)) if matched_pairs else ([], [])

# radar_list,ecg_list=get_matched_filelists(test_wake_radar_dir,test_ecg_dir)
# for i in range(len(radar_list)):
#     print(f'{radar_list[i]},{ecg_list[i]}')#验证是否对应

def loadData(data_size, radar_dir, ecg_dir):
    radar_file_list, ecg_file_list = get_matched_filelists(radar_dir, ecg_dir)
    mmwave_data = []
    ecg_data = []

    # 加载雷达数据并转换为Tensor
    for filename in radar_file_list:
        data = np.load(filename).astype(np.float32)  # 转换为float32
        for i in range(len(data)):
            if i!=len(data)-1:
                data[i]=data[i+1]-data[i]
        k = len(data) // data_size
        for i in range(k):
            slice_data = torch.from_numpy(data[i * data_size: (i + 1) * data_size].copy())  # 转换为Tensor
            mmwave_data.append(slice_data)

    # 加载ECG数据并转换为Tensor
    for filename in ecg_file_list:
        data = np.load(filename).astype(np.float32)  # 转换为float32
        k = len(data) // data_size
        for i in range(k):
            slice_data = torch.from_numpy(data[i * data_size: (i + 1) * data_size].copy())  # 转换为Tensor
            slice_data/=1000
            ecg_data.append(slice_data)
    return ecg_data, ecg_data

# def loadData(data_size, radar_dir, ecg_dir):
#     radar_file_list, ecg_file_list = get_matched_filelists(radar_dir, ecg_dir)
#     mmwave_data = []
#     ecg_data = []
#
#     # 定义归一化函数
#     def minmax_normalize(signal):
#         """将信号归一化到[-1, 1]范围"""
#         s_min = np.min(signal)
#         s_max = np.max(signal)
#         if (s_max - s_min) > 1e-6:  # 防止除零
#             normalized = 2.0 * (signal - s_min) / (s_max - s_min) - 1.0
#         else:
#             normalized = np.zeros_like(signal)  # 处理全零信号
#         return normalized.astype(np.float32)
#
#     # 加载雷达数据并归一化
#     for filename in radar_file_list:
#         data = np.load(filename)
#         for i in range(len(data)):
#             if i!=len(data)-1:
#                 data[i]=data[i+1]-data[i]
#         k = len(data) // data_size
#         for i in range(k):
#             slice_data = data[i*data_size : (i+1)*data_size]
#             normalized_slice = minmax_normalize(slice_data)
#             mmwave_data.append(torch.from_numpy(normalized_slice))
#
#     # 加载ECG数据并归一化（替换原有的/=100操作）
#     for filename in ecg_file_list:
#         data = np.load(filename)
#         k = len(data) // data_size
#         for i in range(k):
#             slice_data = data[i*data_size : (i+1)*data_size]
#             slice_data/=1000
#             normalized_slice = minmax_normalize(slice_data)
#             ecg_data.append(torch.from_numpy(normalized_slice))
#
#     return mmwave_data, ecg_data
# mmWave_data,ecg_data=loadData(1024,train_wake_radar_dir,train_ecg_dir)
# print(np.array(mmWave_data).shape)
# print(np.array(ecg_data).shape)

# def loadData(data_size, radar_dir, ecg_dir):
#     _, ecg_file_list = get_matched_filelists(radar_dir, ecg_dir)
#     mmwave_data = []
#     ecg_data = []
#     # 噪声参数配置
#     noise_intensity = 0.02  # 可调节的噪声强度系数
#     # 加载ECG数据并转换为Tensor
#     for filename in ecg_file_list:
#         data = np.load(filename).astype(np.float32)  # 转换为float32
#         k = len(data) // data_size
#         for i in range(k):
#             # 原始数据切片
#             raw_slice = data[i * data_size: (i + 1) * data_size]
#
#             # 转换为Tensor并标准化
#             ecg_tensor = torch.from_numpy(raw_slice.copy())
#             ecg_tensor = (ecg_tensor - ecg_tensor.mean()) / (ecg_tensor.std() + 1e-8)  # 标准化
#
#             # 生成带噪声的mmwave数据
#             noise = torch.randn_like(ecg_tensor) * noise_intensity
#             mmwave_tensor = ecg_tensor + noise
#
#             # 添加到数据集
#             mmwave_data.append(mmwave_tensor)
#             ecg_data.append(ecg_tensor)
#     return mmwave_data, ecg_data


## 合成数据集
# # 生成对齐1024维度的数据集
# num_samples = 5000
# data, labels = [], []
# for _ in range(num_samples):
#     q, h = generate_synthetic_data()
#     data.append(q)
#     labels.append(h)
#
# dataset = TensorDataset(torch.stack(data), torch.stack(labels))
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
def generate_synthetic_data(seq_length=1024):
    """生成1024长度的合成数据"""
    t = np.linspace(0, 10, seq_length)

    # 心跳信号（1.2Hz基频+谐波）
    heart_signal = 0.5 * np.sin(2 * np.pi * 1.2 * t) + \
                   0.2 * np.sin(4 * np.pi * 1.2 * t)

    # 呼吸信号（0.25Hz低频）
    resp_signal = 1.0 * np.sin(2 * np.pi * 0.25 * t)

    # 混合信号（添加相位调制和噪声）
    mixed_signal = 0.7 * resp_signal + 0.3 * heart_signal + 0.05 * np.random.randn(seq_length)
    return torch.FloatTensor(mixed_signal), torch.FloatTensor(heart_signal)

