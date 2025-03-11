import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
# 加载 .npy 文件
data = np.load("../data/RADAR/angle_fft.npy")

# 打印加载的数组
print(data.shape)


# 3. 如果是一个一维数据，可以直接绘制线形图

plt.plot(data)
plt.title("Plot of 1D Data from angle_fft.npy")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.show()