import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
from mmwave_data_process import loadData
# 加载数据
file1="../data/mmwaveData80/trainData/down_mmwaveData/train_wake/mmwave_0304_crf_0_0_0.npy"
file2="../data/mmwaveData80/trainData/ecgData/ECG_Clean_0304_crf_0_0.npy"
mmfilepath='../LSS_80_down_mmwaveData/mmwave_0307_txc_0_39_0.npy'
ecgfilepath="../data/mmwaveData80/testData/ecgData/ECG_Clean_0307_txc_0_39.npy"
mmwave_data = np.load(mmfilepath)
ecg_data = np.load(ecgfilepath)



# 创建画布和坐标轴
plt.figure(figsize=(12, 6))

# 绘制心电信号（蓝色）
plt.plot(ecg_data, color='blue', alpha=0.7, label='ECG Signal')

# 绘制毫米波信号（橙色）
plt.plot(mmwave_data, color='orange', alpha=0.7, label='mmWave Signal')

# 添加图形元素
plt.title("ECG and mmWave Signal Comparison")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# 自动调整坐标轴范围
plt.xlim(0, max(len(ecg_data), len(mmwave_data)))
plt.ylim(min(np.min(ecg_data), np.min(mmwave_data)),
         max(np.max(ecg_data), np.max(mmwave_data)))

plt.tight_layout()
plt.show()

#
# # 加载 .npy 文件
# data = np.load("../data/mmwaveData80/trainData/down_mmwaveData/train_wake/mmwave_0304_crf_0_0_0.npy")
#
# # 打印加载的数组
# print(data.shape)
#
#
# # 3. 如果是一个一维数据，可以直接绘制线形图
#
# plt.plot(data)
# plt.title("Plot of 1D Data from radar")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.grid(True)
# plt.show()

#
# # # 文件夹路径
# train_wake_radar_dir = "../data/mmwaveData80/trainData/down_mmwaveData/train_wake/"
# train_tired_radar_dir="../data/mmwaveData80/trainData/down_mmwaveData/train_tired/"
# train_ecg_dir = "../data/mmwaveData80/trainData/ecgData"
#
# test_wake_radar_dir = "../data/mmwaveData80/testData/down_mmwaveData/test_wake/"
# test_tired_radar_dir = "../data/mmwaveData80/testData/down_mmwaveData/test_tired"
# test_ecg_dir = "../data/mmwaveData80/testData/ecgData"
# def visualize_comparison(mmwave_data, ecg_data, num_samples=3):
#     """
#     可视化对比带噪声的mmwave数据和原始ECG数据
#     参数：
#         mmwave_data: 带噪声的数据列表
#         ecg_data: 原始数据列表
#         num_samples: 要显示的样本数量
#     """
#     plt.figure(figsize=(12, 6 * num_samples))
#
#     for i in range(min(num_samples, len(mmwave_data))):
#         # 转换为numpy数组（如果数据在GPU上需要先转到CPU）
#         mmwave = mmwave_data[i].cpu().numpy() if mmwave_data[i].is_cuda else mmwave_data[i].numpy()
#         ecg = ecg_data[i].cpu().numpy() if ecg_data[i].is_cuda else ecg_data[i].numpy()
#
#         # 创建时间轴
#         time = np.arange(len(mmwave)) / 100  # 假设采样率100Hz
#
#         # 绘制子图
#         plt.subplot(num_samples, 1, i + 1)
#         plt.plot(time, ecg, label='Original ECG', alpha=0.8, linewidth=1.5)
#         plt.plot(time, mmwave, label='Noisy mmWave', alpha=0.6, linestyle='--')
#         plt.title(f'Sample {i + 1}')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Amplitude')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.show()
#
# # 使用示例：
# # 加载数据
# mmwave, ecg = loadData(1024, train_wake_radar_dir, train_ecg_dir)
#
# # 可视化前3个样本
# visualize_comparison(mmwave, ecg, num_samples=3)