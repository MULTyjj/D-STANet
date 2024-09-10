import numpy as np
import matplotlib.pyplot as plt

pems04_data = np.load('./data/PEMS04/pems04.npz')

# 可以自定义类似的文件包含input + label
print(pems04_data.files)

print(pems04_data['data'].shape)
# (16992, 307, 3)
# 16992 = 59天×24小时×12（每五分钟统计一次流量数据），307为探测器数量，3为特征数。
# 特征：交通流量，平均速度，平均占用率

flow = pems04_data['data'][:, 0, 0]
speed = pems04_data['data'][:, 0, 1]
occupy = pems04_data['data'][:, 0, 2]
fig = plt.figure(figsize=(15, 5))
plt.title('traffic flow in San Francisco')
plt.xlabel('day')
plt.ylabel('flow')
# plt.plot(np.arange(len(flow)), flow, linestyle='-')
plt.plot(np.arange(len(flow)), speed, linestyle='-')
plt.plot(np.arange(len(flow)), occupy, linestyle='-')
fig.autofmt_xdate(rotation=45)  # x轴的刻度标签逆时针旋转45度
plt.show()


# 处理之后的数据
file_data = np.load('./data/PEMS08/PEMS08_r1_d0_w0_dstanet.npz')
print(file_data.files)