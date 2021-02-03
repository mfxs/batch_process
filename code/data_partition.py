# 导入库
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 距离计算
def dist(x1, x2):
    return ((x1 - x2) ** 2).sum()


# 聚类网络
class ClusterNetworks(nn.Module):
    def __init__(self, n_var, n_clu):
        super(ClusterNetworks, self).__init__()
        self.n_var = n_var
        self.n_clu = n_clu
        self.fc1 = nn.Sequential(nn.Linear(self.n_var, 256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128, self.n_clu), nn.ReLU(), nn.Softmax())

    def forward(self, x):
        feature = self.fc1(x)
        feature = self.fc2(feature)
        result = self.fc3(feature)
        return torch.argmax(result, dim=1)


# 自定义损失函数
class MyLoss(nn.Module):
    def __init__(self, mu=0.1):
        super(MyLoss, self).__init__()
        self.mu = mu

    def min_std(self, n):
        seq = torch.tensor(range(n), dtype=torch.float32)
        return torch.std(seq)

    def inner_std(self, sample):
        mean = sample.mean(axis=0)
        return ((sample - mean) ** 2).sum() / sample.shape[0]

    def forward(self, label):
        label_list = torch.unique(label)
        loss = torch.tensor(0.0, requires_grad=True)
        for i in label_list:
            idx = torch.where(label == i)[0]
            sample = data_gpu[label == i]
            loss1 = (torch.std(torch.tensor(idx, dtype=torch.float32)) - self.min_std(idx.shape[0])) / idx.shape[0]
            loss2 = self.inner_std(sample)
            loss = loss + self.mu * loss1 + loss2
        return loss


# 导入数据
data = pd.read_csv('注塑数据/data31.dat', sep='   ', engine='python')
scaler = MinMaxScaler()
data_std = scaler.fit_transform(data)

# 聚类网络构建
n_epoch = 100
data_gpu = torch.tensor(data_std, dtype=torch.float32)
net = ClusterNetworks(n_var=data.shape[1], n_clu=7)
loss = MyLoss(mu=0.1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)

# 训练
net.train()
for epoch in range(n_epoch):
    optimizer.zero_grad()
    y = net(data_gpu)
    loss_ = loss(y)
    loss_.backward()
    optimizer.step()
    print('Epoch:{} Loss:{}'.format(epoch + 1, loss_.item()))

# # 有序层次聚类
# n_cluster = 7
# label = np.array(range(data.shape[0]))
# c = np.unique(label)
# while c.shape[0] > n_cluster:
#     d = []
#     for i in range(c.shape[0] - 1):
#         center1 = data_std[label == c[i]].mean(axis=0)
#         center2 = data_std[label == c[i + 1]].mean(axis=0)
#         d.append(dist(center1, center2))
#     idx = np.argmin(np.array(d))
#     label[label == c[idx + 1]] = c[idx]
#     c = np.unique(label)
#
# # 簇号重新对应
# dict = {}
# for i in range(n_cluster):
#     dict.update({c[i]: i})
# for i in range(label.shape[0]):
#     label[i] = dict[label[i]]
#
# # 画图
# color = ['lightgreen', 'lightcyan', 'violet', 'navajowhite', 'lightcoral', 'lightpink', 'lightyellow']
# plt.plot(data_std)
# for i in np.unique(label):
#     idx = np.where(label == i)
#     plt.fill_betweenx([-10, 10], np.min(idx), np.max(idx), color=color[i])
# plt.show()
