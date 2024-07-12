# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
import numpy as np

# import scipy.io
# data = scipy.io.loadmat('D:\\postgraduate\\CT1\\ZS_TASK1\\101_ZS_task1.mat')
# print(data)
# print("-----------------")
# print(data.keys())


# import scipy.io as sio
# import os
# import numpy as np
# import h5py
# rootPath = 'D:\\postgraduate\\CT1\\ZS_TASK1\\'
# path = 'D:\\postgraduate\\CT1\\ZS_TASK1'
# list = []
# data = L = [[] * 310] * 88
# for file_name in os.listdir(path):
#     # print(file_name)
#     list.append(file_name)
# for i in range(0,88):
#     # print(rootPath+list[i])
#     matdata = h5py.File(rootPath+list[i])
#     arrdata = np.array(matdata['EEG_results']['Power_relative_mean'])
#     arrdataTrans = np.transpose(arrdata)
#     arrdataTransFlat = arrdataTrans.flatten()
#     data[i] = arrdataTransFlat
# for i in range(0,88):
#     x = i + 1
#     print("数据："+str(x))
#     print(data[i])

# matdata = h5py.File('D:\\postgraduate\\CT1\\ZS_TASK1\\101_ZS_task1.mat')
# matdata = h5py.File(rootPath+list[0])
# print('1 ', matdata.keys())
# print('2 ', matdata['EEG_results'])
# print('3 ', matdata['EEG_results']['Power_relative_mean'])
# arrdata = np.array(matdata['EEG_results']['Power_relative_mean'])
# # 进行转置
# arrdataTrans = np.transpose(arrdata)
# # print(arrdataTrans.shape)
# # print(arrdataTrans)
# # 进行展平
# arrdataTransFlat = arrdataTrans.flatten()
# # print(arrdataTransFlat.shape)
# # print(arrdataTransFlat)

# import numpy as np
# import pandas as pd
# y = pd.read_excel('D:/postgraduate/CT1/ZS_task1.xlsx', sheet_name='Sheet1',header=None)
# y = np.array(y).ravel()
# print(y.shape)
# print(y)

# # 行为数据
# import pandas as pd
# import numpy as np
# XBrain = [[] * 10] * 88
# # 视觉追踪数据
# XBrainVision = pd.read_excel('D:/postgraduate/CT1/bevitr.xlsx', sheet_name='Sheet1', header=None)
# XBrainVision = np.array(XBrainVision).ravel()
# # 图片记忆数据
# XBrainPicture = pd.read_excel('D:/postgraduate/CT1/bepire.xlsx', sheet_name='Sheet1', header=None)
# XBrainPicture = np.array(XBrainPicture).ravel()
# # 注意广度数据
# XBrainAttention = pd.read_excel('D:/postgraduate/CT1/beatwi.xlsx', sheet_name='Sheet1', header=None)
# XBrainAttention = np.array(XBrainAttention).ravel()
#
# # 把图片记忆数据划分到三个不同成分的数组
# XBrainPicture1 = np.zeros(88)
# XBrainPicture2 = np.zeros(88)
# XBrainPicture3 = np.zeros(88)
# i1 = 0
# i2 = 0
# i3 = 0
# for i in range(0, 264):
#     if i % 3 == 0:
#         XBrainPicture1[i1] = XBrainPicture[i]
#         i1 = i1 + 1
#     elif i % 3 == 1:
#         XBrainPicture2[i2] = XBrainPicture[i]
#         i2 = i2 + 1
#     else:
#         XBrainPicture3[i3] = XBrainPicture[i]
#         i3 = i3 + 1
# # 把注意广度数据划分到五个不同成分的数组
# XBrainAttention1 = np.zeros(88)
# XBrainAttention2 = np.zeros(88)
# XBrainAttention3 = np.zeros(88)
# XBrainAttention4 = np.zeros(88)
# XBrainAttention5 = np.zeros(88)
# i1 = 0
# i2 = 0
# i3 = 0
# i4 = 0
# i5 = 0
# for i in range(0, 440):
#     if i % 5 == 0:
#         XBrainAttention1[i1] = XBrainAttention[i]
#         i1 = i1 + 1
#     elif i % 5 == 1:
#         XBrainAttention2[i2] = XBrainAttention[i]
#         i2 = i2 + 1
#     elif i % 5 == 2:
#         XBrainAttention3[i3] = XBrainAttention[i]
#         i3 = i3 + 1
#     elif i % 5 == 3:
#         XBrainAttention4[i4] = XBrainAttention[i]
#         i4 = i4 + 1
#     else:
#         XBrainAttention5[i5] = XBrainAttention[i]
#         i5 = i5 + 1
# # 融合数据
# for i in range(0, 88):
#     XBrain[i] = np.hstack((XBrainVision[i], XBrainPicture1[i]))
#     XBrain[i] = np.hstack((XBrain[i], XBrainPicture2[i]))
#     XBrain[i] = np.hstack((XBrain[i], XBrainPicture3[i]))
#     XBrain[i] = np.hstack((XBrain[i], XBrainAttention1[i]))
#     XBrain[i] = np.hstack((XBrain[i], XBrainAttention2[i]))
#     XBrain[i] = np.hstack((XBrain[i], XBrainAttention3[i]))
#     XBrain[i] = np.hstack((XBrain[i], XBrainAttention4[i]))
#     XBrain[i] = np.hstack((XBrain[i], XBrainAttention5[i]))
# print(XBrain)

#     XBrain[i] = np.append(XBrain[i], XBrainVision[i])  # 直接向p_arr里添加p_ #注意一定不要忘记用赋值覆盖原p_arr不然不会变
#     XBrain[i] = np.append(XBrain[i], XBrainPicture[3 * i])
#     XBrain[i] = np.append(XBrain[i], XBrainPicture[3 * i + 1])
#     XBrain[i] = np.append(XBrain[i], XBrainPicture[3 * i + 2])
#     XBrain[i] = np.append(XBrain[i], XBrainAttention[5 * i])
#     XBrain[i] = np.append(XBrain[i], XBrainAttention[5 * i + 1])
#     XBrain[i] = np.append(XBrain[i], XBrainAttention[5 * i + 2])
#     XBrain[i] = np.append(XBrain[i], XBrainAttention[5 * i + 3])
#     XBrain[i] = np.append(XBrain[i], XBrainAttention[5 * i + 4])
# for i in range(0, 88):
#     print(XBrain[i])

# a=[[1,2,3],[4,5,6],[2, 3, 4], [5, 6, 7]]
# a = np.delete(a,2,axis=1)
# print(a)

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
#
# # 准备样本标签和预测概率
# y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 真实标签
# y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5])  # 预测概率
#
# # 计算ROC曲线
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
#
# # 绘制ROC曲线
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

# import torch
#
# # 定义输入特征图
# x = torch.randn(1, 3, 32, 32)
#
# # 使用双线性插值进行上采样
# y = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
#
# # 输出结果
# print(y.shape)

# import random
# from sklearn.neighbors import NearestNeighbors
# import numpy as np
#
#
# class Smote:
#     """
#     SMOTE过采样算法.
#
#
#     Parameters:
#     -----------
#     k: int
#         选取的近邻数目.
#     sampling_rate: int
#         采样倍数, attention sampling_rate < k.
#     newindex: int
#         生成的新样本(合成样本)的索引号.
#     """
#
#     def __init__(self, sampling_rate=5, k=5):
#         self.sampling_rate = sampling_rate
#         self.k = k
#         self.newindex = 0
#
#     def fit(self, X, y):
#         if y is not None:
#             negative_X = X[y == 0]
#             X = X[y == 1]
#
#         n_samples, n_features = X.shape
#         # 初始化一个矩阵, 用来存储合成样本
#         self.synthetic = np.zeros((n_samples * self.sampling_rate, n_features))
#
#         # 找出正样本集(数据集X)中的每一个样本在数据集X中的k个近邻
#         knn = NearestNeighbors(n_neighbors=self.k).fit(X)
#         for i in range(len(X)):
#             k_neighbors = knn.kneighbors(X[i].reshape(1, -1),
#                                          return_distance=False)[0]
#             # 对正样本集(minority class samples)中每个样本, 分别根据其k个近邻生成
#             # sampling_rate个新的样本
#             self.synthetic_samples(X, i, k_neighbors)
#
#         if y is not None:
#             return (np.concatenate((self.synthetic, X, negative_X), axis=0),
#                     np.concatenate(([1] * (len(self.synthetic) + len(X)), y[y == 0]), axis=0))
#
#         return np.concatenate((self.synthetic, X), axis=0)
#
#     # 对正样本集(minority class samples)中每个样本, 分别根据其k个近邻生成sampling_rate个新的样本
#     def synthetic_samples(self, X, i, k_neighbors):
#         for j in range(self.sampling_rate):
#             # 从k个近邻里面随机选择一个近邻
#             neighbor = np.random.choice(k_neighbors)
#             # 计算样本X[i]与刚刚选择的近邻的差
#             diff = X[neighbor] - X[i]
#             # 生成新的数据
#             self.synthetic[self.newindex] = X[i] + random.random() * diff
#             self.newindex += 1
#
#
# X = np.array([[1, 2, 3], [3, 4, 6], [2, 2, 1], [3, 5, 2], [5, 3, 4], [3, 2, 4]])
# y = np.array([1, 1, 1, 0, 0, 0])
# smote = Smote(sampling_rate=2, k=3)
# print(smote.fit(X, y))
