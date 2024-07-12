# 导入必要的库
import math
import random
from numpy import float64
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import h5py
import os
import scipy.io as scio
from scipy.stats import linregress
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

# 创建样本数据
X1 = np.array([[0, 0], [2, 2], [4, 4], [6, 6], [8, 8], [10, 10]])
y1 = np.array([0, 2.3, 4.6, 6.9, 9.2, 11.5])

# 脑电数据
X = [[] * 1500] * 100
# 任务一数据
rootPath = 'D:\\postgraduate\\CT1\\power\\2_ZS\\TASK1\\'
path = 'D:\\postgraduate\\CT1\\power\\2_ZS\\TASK1'
list = []
XTask1 = [[] * 500] * 100
power_fea = np.zeros((100, 9))
power_channel = np.zeros((100, 31))
power = np.zeros((100, 31, 10))
power1 = np.zeros((100, 31, 9))
power9 = np.zeros((100, 9))
for file_name in os.listdir(path):
    # print(file_name)
    list.append(file_name)
for i in range(0, 100):
    # print(rootPath+list[i])
    # matdata = h5py.File(rootPath+list[i])
    # path = r'E:\dataset\data.mat'
    matdata = scio.loadmat(rootPath+list[i])
    arrdataPower = np.array(matdata['EEG_power_results']['Power_relative_mean'])
    arrdataPower = np.transpose(arrdataPower[0][0])
    # print(arrdataPower)
    # print(arrdataPower.shape)
    # print(arrdataPower[0][0]) 这个才是对的
    # Power_relative_mean 平均相对功率已经是归一化了的 其他的没有
    arrdataTransPower = np.transpose(arrdataPower)
    # print(arrdataTransPower)
    # 取出Power_relative_mean的每一个通道对应的每一个特征值，总共31*10=310个
    power[i] = arrdataTransPower

    # 取出Power_relative_mean的31个通道数组
    for m in range(0, 31):
        for j in range(0, 9):
            power_channel[i][m] = power_channel[i][m] + arrdataPower[j][m]

    # 取出Power_relative_mean的九个特征值数组
    for m in range(0, 9):
        for j in range(0, 31):
            power_fea[i][m] = power_fea[i][m] + arrdataTransPower[j][m]
        # print(power_1fea[i][m])
    arrdataTransFlatPower = arrdataTransPower.flatten()

    arrdataR1 = np.array(matdata['EEG_power_results']['R1_mean'])
    arrdataR1 = arrdataR1[0][0]
    arrdataTransR1 = np.transpose(arrdataR1)
    arrdataTransFlatR1 = arrdataTransR1.flatten()
    # print(arrdataTransFlatR1)

    arrdataR2 = np.array(matdata['EEG_power_results']['R2_mean'])
    arrdataR2 = arrdataR2[0][0]
    arrdataTransR2 = np.transpose(arrdataR2)
    arrdataTransFlatR2 = arrdataTransR2.flatten()

    arrdataR3 = np.array(matdata['EEG_power_results']['R3_mean'])
    arrdataR3 = arrdataR3[0][0]
    arrdataTransR3 = np.transpose(arrdataR3)
    arrdataTransFlatR3 = arrdataTransR3.flatten()
    arrdataR4 = np.array(matdata['EEG_power_results']['R4_mean'])
    arrdataR4 = arrdataR4[0][0]
    arrdataTransR4 = np.transpose(arrdataR4)
    arrdataTransFlatR4 = arrdataTransR4.flatten()
    arrdataR5 = np.array(matdata['EEG_power_results']['R5_mean'])
    arrdataR5 = arrdataR5[0][0]
    arrdataTransR5 = np.transpose(arrdataR5)
    arrdataTransFlatR5 = arrdataTransR5.flatten()
    arrdataR6 = np.array(matdata['EEG_power_results']['R6_mean'])
    arrdataR6 = arrdataR6[0][0]
    arrdataTransR6 = np.transpose(arrdataR6)
    arrdataTransFlatR6 = arrdataTransR6.flatten()
    # arrdataAll = np.hstack((arrdataTransFlatPower, arrdataTransFlatR1))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR2))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR3))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR4))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR5))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR6))
    XTask1[i] = arrdataTransFlatPower
# 任务二数据
rootPath2 = 'D:\\postgraduate\\CT1\\power\\2_ZS\\TASK2\\'
path2 = 'D:\\postgraduate\\CT1\\power\\2_ZS\\TASK2'
list2 = []
XTask2 = [[] * 500] * 100
for file_name in os.listdir(path2):
    # print(file_name)
    list2.append(file_name)
for i in range(0, 100):
    # print(rootPath+list[i])
    # matdata = h5py.File(rootPath2+list2[i])
    # arrdataPower = np.array(matdata['EEG_power_results']['Power_relative_mean'])
    matdata = scio.loadmat(rootPath2 + list2[i])
    arrdataPower = np.array(matdata['EEG_power_results']['Power_relative_mean'])
    arrdataPower = np.transpose(arrdataPower[0][0])

    arrdataTransPower = np.transpose(arrdataPower)

    # 取出Power_relative_mean的每一个通道对应的每一个特征值，总共31*10=310个
    power[i] = np.add(power[i],arrdataTransPower)

    # 取出Power_relative_mean的31个通道数组
    for m in range(0, 31):
        for j in range(0, 9):
            power_channel[i][m] = power_channel[i][m] + arrdataPower[j][m]

    # 取出Power_relative_mean的九个特征值数组
    for m in range(0, 9):
        for j in range(0, 31):
            power_fea[i][m] = power_fea[i][m] + arrdataTransPower[j][m]
    arrdataTransFlatPower = arrdataTransPower.flatten()

    arrdataR1 = np.array(matdata['EEG_power_results']['R1_mean'])
    arrdataR1 = arrdataR1[0][0]
    arrdataTransR1 = np.transpose(arrdataR1)
    arrdataTransFlatR1 = arrdataTransR1.flatten()
    # print(arrdataTransFlatR1)

    arrdataR2 = np.array(matdata['EEG_power_results']['R2_mean'])
    arrdataR2 = arrdataR2[0][0]
    arrdataTransR2 = np.transpose(arrdataR2)
    arrdataTransFlatR2 = arrdataTransR2.flatten()
    arrdataR3 = np.array(matdata['EEG_power_results']['R3_mean'])
    arrdataR3 = arrdataR3[0][0]
    arrdataTransR3 = np.transpose(arrdataR3)
    arrdataTransFlatR3 = arrdataTransR3.flatten()
    arrdataR4 = np.array(matdata['EEG_power_results']['R4_mean'])
    arrdataR4 = arrdataR4[0][0]
    arrdataTransR4 = np.transpose(arrdataR4)
    arrdataTransFlatR4 = arrdataTransR4.flatten()
    arrdataR5 = np.array(matdata['EEG_power_results']['R5_mean'])
    arrdataR5 = arrdataR5[0][0]
    arrdataTransR5 = np.transpose(arrdataR5)
    arrdataTransFlatR5 = arrdataTransR5.flatten()
    arrdataR6 = np.array(matdata['EEG_power_results']['R6_mean'])
    arrdataR6 = arrdataR6[0][0]
    arrdataTransR6 = np.transpose(arrdataR6)
    arrdataTransFlatR6 = arrdataTransR6.flatten()
    # arrdataAll = np.hstack((arrdataTransFlatPower, arrdataTransFlatR1))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR2))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR3))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR4))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR5))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR6))
    XTask2[i] = arrdataTransFlatPower
# 任务三数据
rootPath3 = 'D:\\postgraduate\\CT1\\power\\2_ZS\\TASK3\\'
path3 = 'D:\\postgraduate\\CT1\\power\\2_ZS\\TASK3'
list3 = []
XTask3 = [[] * 500] * 100
for file_name in os.listdir(path3):
    # print(file_name)
    list3.append(file_name)
for i in range(0, 100):
    # print(rootPath+list[i])
    # matdata = h5py.File(rootPath3+list3[i])
    # arrdataPower = np.array(matdata['EEG_power_results']['Power_relative_mean'])
    matdata = scio.loadmat(rootPath3 + list3[i])
    arrdataPower = np.array(matdata['EEG_power_results']['Power_relative_mean'])
    arrdataPower = np.transpose(arrdataPower[0][0])

    arrdataTransPower = np.transpose(arrdataPower)

    # 取出Power_relative_mean的每一个通道对应的每一个特征值，总共31*10=310个
    power[i] = np.add(power[i], arrdataTransPower)

    # 取出Power_relative_mean的31个通道数组
    for m in range(0, 31):
        for j in range(0, 9):
            power_channel[i][m] = power_channel[i][m] + arrdataPower[j][m]

    # 取出Power_relative_mean的九个特征值数组
    for m in range(0, 9):
        for j in range(0, 31):
            power_fea[i][m] = power_fea[i][m] + arrdataTransPower[j][m]
    arrdataTransFlatPower = arrdataTransPower.flatten()

    arrdataR1 = np.array(matdata['EEG_power_results']['R1_mean'])
    arrdataR1 = arrdataR1[0][0]
    arrdataTransR1 = np.transpose(arrdataR1)
    arrdataTransFlatR1 = arrdataTransR1.flatten()

    arrdataR2 = np.array(matdata['EEG_power_results']['R2_mean'])
    arrdataR2 = arrdataR2[0][0]
    arrdataTransR2 = np.transpose(arrdataR2)
    arrdataTransFlatR2 = arrdataTransR2.flatten()
    arrdataR3 = np.array(matdata['EEG_power_results']['R3_mean'])
    arrdataR3 = arrdataR3[0][0]
    arrdataTransR3 = np.transpose(arrdataR3)
    arrdataTransFlatR3 = arrdataTransR3.flatten()
    arrdataR4 = np.array(matdata['EEG_power_results']['R4_mean'])
    arrdataR4 = arrdataR4[0][0]
    arrdataTransR4 = np.transpose(arrdataR4)
    arrdataTransFlatR4 = arrdataTransR4.flatten()
    arrdataR5 = np.array(matdata['EEG_power_results']['R5_mean'])
    arrdataR5 = arrdataR5[0][0]
    arrdataTransR5 = np.transpose(arrdataR5)
    arrdataTransFlatR5 = arrdataTransR5.flatten()
    arrdataR6 = np.array(matdata['EEG_power_results']['R6_mean'])
    arrdataR6 = arrdataR6[0][0]
    arrdataTransR6 = np.transpose(arrdataR6)
    arrdataTransFlatR6 = arrdataTransR6.flatten()
    # arrdataAll = np.hstack((arrdataTransFlatPower, arrdataTransFlatR1))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR2))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR3))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR4))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR5))
    # arrdataAll = np.hstack((arrdataAll, arrdataTransFlatR6))
    XTask3[i] = arrdataTransFlatPower

# 去除第10列的数据 没有参考价值
for i in range(0, 100):
    power1[i] = np.delete(power[i], 9, axis=1)

# 获取第10行，第1列 第14行，第1列 第14行，第2列 第26行，第1列 第27行，第1列...的数据
for i in range(0, 100):
    m = 0
    for j in range(0, 31):
        for k in range(0, 9):
            if (j == 9 and k == 0) or (j == 13 and k == 0) or (j == 13 and k == 1) or (j == 14 and k == 0) or (
                    j == 15 and k == 0) or (j == 16 and k == 0) or (j == 18 and k == 0) or (j == 25 and k == 0) or (
                    j == 26 and k == 0):
            # if (j==9 and k==0) or (j==13 and k==0) or (j==13 and k==1) or (j==25 and k==0) or (j==26 and k==0) or (j==0 and k==8) or (j==14 and k==0) or (j==19 and k==8) or (j==29 and k==0):
                power9[i][m] = power1[i][j][k]
                m = m + 1
print("打印power9的数据：")
print(power9)

# print("power_fea:")
# print(power_fea)
# print("power_channel:")
# for i in range(0, 88):
#     print(power_channel[i])

# 融合三个任务数据
for i in range(0, 100):
    X[i] = np.add(XTask1[i], XTask2[i])
    X[i] = np.add(X[i], XTask3[i])
# print(X[0])


# # 行为数据
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
# # 把图片记忆数据划分到三个不同成分的数组
# # meanRT_ByType
# XBrainPicture1 = np.zeros(88)
# # countOfCorrectButton_ByType_Corrected
# XBrainPicture2 = np.zeros(88)
# # countOfCorrectButton_ByType
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
# # 归一化
# max1 = XBrainPicture1.max()
# min1 = XBrainPicture1.min()
# max2 = XBrainPicture2.max()
# min2 = XBrainPicture2.min()
# max3 = XBrainPicture3.max()
# min3 = XBrainPicture3.min()
# for i in range(0, 88):
#     XBrainPicture1[i] = (XBrainPicture1[i]-min1)/(max1-min1)
#     XBrainPicture2[i] = (XBrainPicture2[i] - min2) / (max2 - min2)
#     XBrainPicture3[i] = (XBrainPicture3[i] - min3) / (max3 - min3)
#
# # 把注意广度数据划分到五个不同成分的数组
# # medianRTButton_ByType_Corrected
# XBrainAttention1 = np.zeros(88)
# # meanRT_ByType
# XBrainAttention2 = np.zeros(88)
# # percentageCorrectButton_ByType
# XBrainAttention3 = np.zeros(88)
# #countOfCorrectButton_ByType_Corrected
# XBrainAttention4 = np.zeros(88)
# # countOfCorrectButton_ByType
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
# # 归一化
# max1 = XBrainAttention1.max()
# min1 = XBrainAttention1.min()
# max2 = XBrainAttention2.max()
# min2 = XBrainAttention2.min()
# max3 = XBrainAttention3.max()
# min3 = XBrainAttention3.min()
# max4 = XBrainAttention4.max()
# min4 = XBrainAttention4.min()
# max5 = XBrainAttention5.max()
# min5 = XBrainAttention5.min()
# for i in range(0, 88):
#     XBrainAttention1[i] = (XBrainAttention1[i] - min1) / (max1 - min1)
#     XBrainAttention2[i] = (XBrainAttention2[i] - min2) / (max2 - min2)
#     XBrainAttention3[i] = (XBrainAttention3[i] - min3) / (max3 - min3)
#     XBrainAttention4[i] = (XBrainAttention4[i] - min4) / (max4 - min4)
#     XBrainAttention5[i] = (XBrainAttention5[i] - min5) / (max5 - min5)
# # 融合数据
# for i in range(0, 88):
#     XBrain[i] = np.hstack((XBrainVision[i], XBrainPicture3[i]))
#     # XBrain[i] = np.hstack((XBrain[i], XBrainPicture2[i]))
#     # XBrain[i] = np.hstack((XBrain[i], XBrainPicture3[i]))
#     # XBrain[i] = np.hstack((XBrain[i], XBrainAttention1[i]))
#     # XBrain[i] = np.hstack((XBrain[i], XBrainAttention2[i]))
#     # XBrain[i] = np.hstack((XBrain[i], XBrainAttention3[i]))
#     # XBrain[i] = np.hstack((XBrain[i], XBrainAttention4[i]))
#     XBrain[i] = np.hstack((XBrain[i], XBrainAttention5[i]))

# 完成度（绩效）
y = pd.read_excel('D:/postgraduate/CT1/ZS.xlsx', sheet_name='Sheet1', header=None)
y = np.array(y).ravel()

# 归一化
# for i in range(0, 100):
#     y[i] = (y[i]-0.2)/2.8

# 进行smote采样
class Smote:
    """
    SMOTE过采样算法.


    Parameters:
    -----------
    k: int
        选取的近邻数目.
    sampling_rate: int
        采样倍数, attention sampling_rate < k.
    newindex: int
        生成的新样本(合成样本)的索引号.
    """

    def __init__(self, sampling_rate=5, k=5):
        self.sampling_rate = sampling_rate
        self.k = k
        self.newindex = 0

    def fit(self, X, y=None):
        if y is not None:
            negative_X = X[y == 0]
            X = X[y == 1]

        n_samples, n_features = X.shape
        # 初始化一个矩阵, 用来存储合成样本
        self.synthetic = np.zeros((n_samples * self.sampling_rate, n_features))

        # 找出正样本集(数据集X)中的每一个样本在数据集X中的k个近邻
        knn = NearestNeighbors(n_neighbors=self.k).fit(X)
        for i in range(len(X)):
            k_neighbors = knn.kneighbors(X[i].reshape(1, -1),
                                         return_distance=False)[0]
            # 对正样本集(minority class samples)中每个样本, 分别根据其k个近邻生成
            # sampling_rate个新的样本
            self.synthetic_samples(X, i, k_neighbors)

        if y is not None:
            return (np.concatenate((self.synthetic, X, negative_X), axis=0),
                    np.concatenate(([1] * (len(self.synthetic) + len(X)), y[y == 0]), axis=0))

        return np.concatenate((self.synthetic, X), axis=0)

    # 对正样本集(minority class samples)中每个样本, 分别根据其k个近邻生成sampling_rate个新的样本
    def synthetic_samples(self, X, i, k_neighbors):
        for j in range(self.sampling_rate):
            # 从k个近邻里面随机选择一个近邻
            neighbor = np.random.choice(k_neighbors)
            # 计算样本X[i]与刚刚选择的近邻的差
            diff = X[neighbor] - X[i]
            # 生成新的数据
            self.synthetic[self.newindex] = X[i] + random.random() * diff
            self.newindex += 1

smote_power9_y = np.zeros((100, 10))
for i in range(0, 100):
    for j in range(0, 9):
        smote_power9_y[i][j] = power9[i][j]
    smote_power9_y[i][9] = y[i]
smote = Smote(sampling_rate=1, k=5)
new_smote_power9 = smote.fit(smote_power9_y)
smote_power9 = np.zeros((200, 9))
smote_y = np.zeros(200)
for i in range(0, 200):
    for j in range(0, 9):
        smote_power9[i][j] = new_smote_power9[i][j]
    smote_y[i] = new_smote_power9[i][9]

# 输出为EXCEL文件
# import xlwt
# workbook = xlwt.Workbook()
# sheet = workbook.add_sheet("Sheet")
#
# for i in range(len(new_smote_power9)):
#     for j in range(len(new_smote_power9[i])):
#         sheet.write(i, j, new_smote_power9[i][j])
#
# workbook.save("smote_power9_y.xls")
# print("完成")

# 将数据集划分为训练集和测试集（留出法）
X_train, X_test, y_train, y_test = train_test_split(smote_power9, smote_y, test_size=0.3, random_state=40)

# 定义需要使用的SVM模型
svr = SVR()

# 创建3折交叉验证对象
kf = KFold(n_splits=3)
# 定义超参数空间和网格搜索对象
# 超参数自动优化方法
# kernel:线性核函数，径向基核函数，多项式核函数 gamma:多项式的系数[0.001, 0.01, 0.1, 1, 10, 100]['scale', 'auto'] 'C'（正则化参数）:[0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
param_grid = {'kernel': ['linear', 'rbf', 'poly'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(svr, param_grid, cv=kf)
# print(555)
# 在训练集上使用网格搜索进行超参数寻优
grid_search.fit(X_train, y_train)
# print(111)
# 输出最佳超参数组合和在测试集上的性能指标
print("Best parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
print("Test score: %.2f" % grid_search.score(X_test, y_test))
# print(222)
# Best parameters: {'C': 0.6, 'gamma': 'scale', 'kernel': 'rbf'}
# Test score: -0.20
#优化后的基模型
# print(grid_search.best_params_['kernel'])
model_svr = SVR(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'], gamma=grid_search.best_params_['gamma'])
model_svr.fit(X_train, y_train)
predict_results = model_svr.predict(X_test)


# print("y_test:")
# print(y_test)
# print("predict_results:")
# print(predict_results)

# # 计算MSE均方误差
# sum = 0
# MSE = 0
# for i in range(0, 27):
#     sum = predict_results[i] - y_test[i]
#     sum = sum * sum
#     MSE = MSE + sum
# MSE = MSE/27
# print(f"均方误差MSE:{MSE}")

# # 计算MAE平均绝对误差
# def calculate_the_MAE(predicted_data,actual_data):
#     # 定义一个变量用于存储所有样本的绝对误差之和
#     the_sum_of_error = 0
#     # 开始逐渐遍历每一个样本
#     for i in range(len(actual_data)):
#         # 不断累加求和，计算所有样本的绝对误差之和
#         the_sum_of_error += abs(predicted_data[i]-actual_data[i])
#     # 计算所有样本的平均绝对误差
#     MAE = the_sum_of_error/float(len(actual_data))
#     return MAE
# Mean_Absolute_Error = calculate_the_MAE(predict_results,y_test)
# print(f"平均绝对误差MAE:{Mean_Absolute_Error}")

# 计算指标
mae = mean_absolute_error(y_test, predict_results)
mse = mean_squared_error(y_test, predict_results)
rmse = np.sqrt(mean_squared_error(y_test, predict_results))
print("平均绝对误差MAE:", mae)
print("均方误差MSE:", mse)
print("均方根误差RMSE:", rmse)

# 计算相关系数R值函数
def get_r(x, y):  # R
    if len(x) == len(y):
        n = len(x)
        sum_xy = np.sum(np.sum(x * y))
        sum_x = np.sum(np.sum(x))
        sum_y = np.sum(np.sum(y))
        sum_x2 = np.sum(np.sum(x * x))
        sum_y2 = np.sum(np.sum(y * y))
        pc = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        return pc
    else:
        return None

# 计算power_relative_mean特征相关性 绘制散点图
# power_feaTrans = np.transpose(power_fea)
# for i in range(0, 9):
#     r = get_r(power_feaTrans[i], y)
#     print(f"第{i+1}维度R值为:{r}")
#     plt.scatter(power_feaTrans[i], y)
#     plt.xlabel(f'power_{i+1}fea')
#     plt.ylabel('y')
#     plt.show()

# power_feaTrans = np.transpose(power_fea)
# for i in range(0, 9):
#     r = get_r(power_feaTrans[i], y)
#     slope, intercept, r_value, p_value, std_err = linregress(power_feaTrans[i], y)
#     print("斜率：", slope)
#     print("截距：", intercept)
#     print("相关系数：", r_value)
#     print("R值：", r)
#     print("p 值：", p_value)
#     print("标准误差：", std_err)
#     plt.scatter(power_feaTrans[i], y)
#     plt.xlabel(f'power_{i+1}fea')
#     plt.ylabel('y')
#     plt.show()

# 计算power_relative_mean通道相关性 绘制散点图
# power_channelTrans = np.transpose(power_channel)
# for i in range(0, 31):
#     r = get_r(power_channelTrans[i], y)
#     print(f"第{i+1}通道R值为:{r}")
#     plt.scatter(power_channelTrans[i], y)
#     plt.xlabel(f'power_{i+1}channel')
#     plt.ylabel('y')
#     plt.show()

# 计算皮尔森相关系数
# PearsonRResult(statistic=0.3212320605558783, pvalue=0.10229614709321634)p值大于0.05，线性关系不显著
# print(len(predict_results))#27

# 计算power_relative_mean所有的点相关性
# x_power = np.zeros(100)
# for j in range(0, 31):
#     for k in range(0, 9):
#         for i in range(0, 100):
#             x_power[i] = power1[i][j][k]
#         r = get_r(x_power, y)
#         # print(f"相关性为：{r}")
#         if r >= 0.18:
#             print(f"第{j+1}行，第{k+1}列具有相关性,其相关性为：{r}")

# 计算R1所有的点相关性
# for i in range(0, 31):
#     re_R1 = np.transpose(X)
#     r = get_r(re_R1[i], y)
#     if r >= 0.2:
#         print(i)
#         print(r)


# 计算那9个点之和的相关性
# power91 = np.zeros(100)
# for i in range(0, 100):
#     power91[i] = power9[i][0]+power9[i][1]+power9[i][2]+power9[i][3]+power9[i][4]+power9[i][5]+power9[i][6]+power9[i][7]+power9[i][8]
# rela9 = get_r(power91, y)
# print(f"那9个点和的相关性为：{rela9}")


dat = []
for i in range(1, 31):
    dat.append(i)
# plt.xlim(0, 160)  # 限定横轴的范围
palette = pyplot.get_cmap('Set1')
# plt.scatter(y_test, predict_results)
plt.plot(dat, y_test, color=palette(3), marker='*', label='True')
plt.plot(dat, predict_results, color=palette(1), marker='^', label='Predict')
plt.xticks(size=15)
plt.yticks(size=15)
plt.legend()  # 让图例生效
# plt.show()
plt.show()