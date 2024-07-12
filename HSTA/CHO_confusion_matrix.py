from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# 5:92,6:89,7:48,9:78,10:79,11:82,12:83,13:78

title = ""
best_accuracy_list = []
for num in range(5, 6):
    # 从左手特征文件夹和标签文件夹加载数据
    feature_folder = "D:\postgraduate\MIdata/52_subject\left_feature_DE/"
    label_folder = "D:\postgraduate\MIdata/52_subject\left_label/"

    # # 从左手特征文件夹中加载数据
    left_features = []
    test_left_features = []
    for i in range(1, 53):
        if i == num:
            test_feature_file = np.load(feature_folder + "S{:02}.npy".format(i))
            test_left_features.append(test_feature_file)
            continue
        feature_file = np.load(feature_folder + "S{:02}.npy".format(i))
        if i == 7 or i == 9 or i == 46:
            feature_file = feature_file[:100, :, :, :]
        # print(feature_file.shape)
        left_features.append(feature_file)

    # # # 从左手标签文件夹中加载数据
    left_labels = []
    test_left_labels = []
    for i in range(1, 53):
        if i == num:
            test_label_file = np.load(label_folder + "left_label{:02}.npy".format(i))
            test_left_labels.append(test_label_file)
            continue
        label_file = np.load(label_folder + "left_label{:02}.npy".format(i))
        if i == 7 or i == 9 or i == 46:
            label_file = label_file[:100]
        # print(label_file.shape)
        left_labels.append(label_file)
    #
    left_features = np.array(left_features)
    left_labels = np.array(left_labels)
    left_features = left_features.reshape(-1, 64, 5, 3)
    left_labels = left_labels.reshape(-1)
    test_left_features = np.array(test_left_features)
    test_left_labels = np.array(test_left_labels)
    test_left_features = test_left_features.reshape(-1, 64, 5, 3)
    test_left_labels = test_left_labels.reshape(-1)
    # print(left_features.shape)  # (5200, 64, 5, 3)
    # 接下来要把64个通道分为5个区域 基于国际10-10系统的64通道蒙太奇脑电图技术
    # 定义5个区域
    left_features_partition1 = []  # 1-3,33-37
    left_features_partition2 = []  # 4-11,38-47
    left_features_partition3 = []  # 12-15,48-52
    left_features_partition4 = []  # 16-24,31-32,53-61
    left_features_partition5 = []  # 25-30,62-64

    left_features_partition1_1 = left_features[:, 0:3, :, :]
    left_features_partition1_2 = left_features[:, 32:37, :, :]
    left_features_partition1 = np.concatenate((left_features_partition1_1, left_features_partition1_2), axis=1)
    # print(left_features_partition1.shape)

    left_features_partition2_1 = left_features[:, 3:11, :, :]
    left_features_partition2_2 = left_features[:, 37:47, :, :]
    left_features_partition2 = np.concatenate((left_features_partition2_1, left_features_partition2_2), axis=1)
    # print(left_features_partition2.shape)

    left_features_partition3_1 = left_features[:, 11:15, :, :]
    left_features_partition3_2 = left_features[:, 47:52, :, :]
    left_features_partition3 = np.concatenate((left_features_partition3_1, left_features_partition3_2), axis=1)
    # print(left_features_partition3.shape)

    left_features_partition4_1 = left_features[:, 15:24, :, :]
    left_features_partition4_2 = left_features[:, 30:32, :, :]
    left_features_partition4_3 = left_features[:, 52:61, :, :]
    left_features_partition4 = np.concatenate((left_features_partition4_1, left_features_partition4_2, left_features_partition4_3), axis=1)
    # print(left_features_partition4.shape)

    left_features_partition5_1 = left_features[:, 24:30, :, :]
    left_features_partition5_2 = left_features[:, 61:64, :, :]
    left_features_partition5 = np.concatenate((left_features_partition5_1, left_features_partition5_2), axis=1)
    # print(left_features_partition5.shape)

    # 测试集
    # 定义5个区域
    test_left_features_partition1 = []  # 1-3,33-37
    test_left_features_partition2 = []  # 4-11,38-47
    test_left_features_partition3 = []  # 12-15,48-52
    test_left_features_partition4 = []  # 16-24,31-32,53-61
    test_left_features_partition5 = []  # 25-30,62-64

    test_left_features_partition1_1 = test_left_features[:, 0:3, :, :]
    test_left_features_partition1_2 = test_left_features[:, 32:37, :, :]
    test_left_features_partition1 = np.concatenate((test_left_features_partition1_1, test_left_features_partition1_2), axis=1)
    # print(left_features_partition1.shape)

    test_left_features_partition2_1 = test_left_features[:, 3:11, :, :]
    test_left_features_partition2_2 = test_left_features[:, 37:47, :, :]
    test_left_features_partition2 = np.concatenate((test_left_features_partition2_1, test_left_features_partition2_2), axis=1)
    # print(left_features_partition2.shape)

    test_left_features_partition3_1 = test_left_features[:, 11:15, :, :]
    test_left_features_partition3_2 = test_left_features[:, 47:52, :, :]
    test_left_features_partition3 = np.concatenate((test_left_features_partition3_1, test_left_features_partition3_2), axis=1)
    # print(left_features_partition3.shape)

    test_left_features_partition4_1 = test_left_features[:, 15:24, :, :]
    test_left_features_partition4_2 = test_left_features[:, 30:32, :, :]
    test_left_features_partition4_3 = test_left_features[:, 52:61, :, :]
    test_left_features_partition4 = np.concatenate(
        (test_left_features_partition4_1, test_left_features_partition4_2, test_left_features_partition4_3), axis=1)
    # print(left_features_partition4.shape)

    test_left_features_partition5_1 = test_left_features[:, 24:30, :, :]
    test_left_features_partition5_2 = test_left_features[:, 61:64, :, :]
    test_left_features_partition5 = np.concatenate((test_left_features_partition5_1, test_left_features_partition5_2), axis=1)
    # print(left_features_partition5.shape)

    # print(left_labels.shape)  # (5200)
    # print(left_labels)  # (5200)

    # print("--------------------------------------------------------------------------")

    # # --------------------------------------------------------------------------
    #
    # 从右手特征文件夹和标签文件夹加载数据
    feature_folder = "D:\postgraduate\MIdata/52_subject/right_feature_DE/"
    label_folder = "D:\postgraduate\MIdata/52_subject/right_label/"

    # 从右手特征文件夹中加载数据
    right_features = []
    test_right_features = []
    for i in range(1, 53):
        if i == num:
            test_feature_file = np.load(feature_folder + "S{:02}.npy".format(i))
            test_right_features.append(test_feature_file)
            continue
        feature_file = np.load(feature_folder + "S{:02}.npy".format(i))
        if i == 7 or i == 9 or i == 46:
            feature_file = feature_file[:100, :, :, :]
        # print(feature_file.shape)
        right_features.append(feature_file)

    # 从右手标签文件夹中加载数据
    right_labels = []
    test_right_labels = []
    for i in range(1, 53):
        if i == num:
            test_label_file = np.load(label_folder + "right_label{:02}.npy".format(i))
            test_right_labels.append(test_label_file)
            continue
        label_file = np.load(label_folder + "right_label{:02}.npy".format(i))
        if i == 7 or i == 9 or i == 46:
            label_file = label_file[:100]
        # print(label_file.shape)
        right_labels.append(label_file)

    right_features = np.array(right_features)
    right_labels = np.array(right_labels)
    right_features = right_features.reshape(-1, 64, 5, 3)
    right_labels = right_labels.reshape(-1)
    test_right_features = np.array(test_right_features)
    test_right_labels = np.array(test_right_labels)
    test_right_features = test_right_features.reshape(-1, 64, 5, 3)
    test_right_labels = test_right_labels.reshape(-1)
    # print(right_features.shape)  # (5200, 64, 5, 3)

    # 接下来要把64个通道分为5个区域 基于国际10-10系统的64通道蒙太奇脑电图技术
    # 定义5个区域
    right_features_partition1 = []  # 1-3,33-37
    right_features_partition2 = []  # 4-11,38-47
    right_features_partition3 = []  # 12-15,48-52
    right_features_partition4 = []  # 16-24,31-32,53-61
    right_features_partition5 = []  # 25-30,62-64

    right_features_partition1_1 = right_features[:, 0:3, :, :]
    right_features_partition1_2 = right_features[:, 32:37, :, :]
    right_features_partition1 = np.concatenate((right_features_partition1_1, right_features_partition1_2), axis=1)
    # print(right_features_partition1.shape)

    right_features_partition2_1 = right_features[:, 3:11, :, :]
    right_features_partition2_2 = right_features[:, 37:47, :, :]
    right_features_partition2 = np.concatenate((right_features_partition2_1, right_features_partition2_2), axis=1)
    # print(right_features_partition2.shape)

    right_features_partition3_1 = right_features[:, 11:15, :, :]
    right_features_partition3_2 = right_features[:, 47:52, :, :]
    right_features_partition3 = np.concatenate((right_features_partition3_1, right_features_partition3_2), axis=1)
    # print(right_features_partition3.shape)

    right_features_partition4_1 = right_features[:, 15:24, :, :]
    right_features_partition4_2 = right_features[:, 30:32, :, :]
    right_features_partition4_3 = right_features[:, 52:61, :, :]
    right_features_partition4 = np.concatenate((right_features_partition4_1, right_features_partition4_2, right_features_partition4_3), axis=1)
    # print(right_features_partition4.shape)

    right_features_partition5_1 = right_features[:, 24:30, :, :]
    right_features_partition5_2 = right_features[:, 61:64, :, :]
    right_features_partition5 = np.concatenate((right_features_partition5_1, right_features_partition5_2), axis=1)
    # print(right_features_partition5.shape)

    # 测试集
    # 定义5个区域
    test_right_features_partition1 = []  # 1-3,33-37
    test_right_features_partition2 = []  # 4-11,38-47
    test_right_features_partition3 = []  # 12-15,48-52
    test_right_features_partition4 = []  # 16-24,31-32,53-61
    test_right_features_partition5 = []  # 25-30,62-64

    test_right_features_partition1_1 = test_right_features[:, 0:3, :, :]
    test_right_features_partition1_2 = test_right_features[:, 32:37, :, :]
    test_right_features_partition1 = np.concatenate((test_right_features_partition1_1, test_right_features_partition1_2), axis=1)
    # print(right_features_partition1.shape)

    test_right_features_partition2_1 = test_right_features[:, 3:11, :, :]
    test_right_features_partition2_2 = test_right_features[:, 37:47, :, :]
    test_right_features_partition2 = np.concatenate((test_right_features_partition2_1, test_right_features_partition2_2), axis=1)
    # print(right_features_partition2.shape)

    test_right_features_partition3_1 = test_right_features[:, 11:15, :, :]
    test_right_features_partition3_2 = test_right_features[:, 47:52, :, :]
    test_right_features_partition3 = np.concatenate((test_right_features_partition3_1, test_right_features_partition3_2), axis=1)
    # print(right_features_partition3.shape)

    test_right_features_partition4_1 = test_right_features[:, 15:24, :, :]
    test_right_features_partition4_2 = test_right_features[:, 30:32, :, :]
    test_right_features_partition4_3 = test_right_features[:, 52:61, :, :]
    test_right_features_partition4 = np.concatenate(
        (test_right_features_partition4_1, test_right_features_partition4_2, test_right_features_partition4_3), axis=1)
    # print(right_features_partition4.shape)

    test_right_features_partition5_1 = test_right_features[:, 24:30, :, :]
    test_right_features_partition5_2 = test_right_features[:, 61:64, :, :]
    test_right_features_partition5 = np.concatenate((test_right_features_partition5_1, test_right_features_partition5_2), axis=1)
    # print(right_features_partition5.shape)

    # print(right_labels.shape)  # (5200)
    # print(right_labels)  # (5200)

    #
    # # --------------------------------------------------------------------------
    # 分别把不同脑区的左右手拼接起来
    features_partition1 = np.concatenate((left_features_partition1, right_features_partition1), axis=0)
    features_partition2 = np.concatenate((left_features_partition2, right_features_partition2), axis=0)
    features_partition3 = np.concatenate((left_features_partition3, right_features_partition3), axis=0)
    features_partition4 = np.concatenate((left_features_partition4, right_features_partition4), axis=0)
    features_partition5 = np.concatenate((left_features_partition5, right_features_partition5), axis=0)
    # 测试集
    test_features_partition1 = np.concatenate((test_left_features_partition1, test_right_features_partition1), axis=0)
    test_features_partition2 = np.concatenate((test_left_features_partition2, test_right_features_partition2), axis=0)
    test_features_partition3 = np.concatenate((test_left_features_partition3, test_right_features_partition3), axis=0)
    test_features_partition4 = np.concatenate((test_left_features_partition4, test_right_features_partition4), axis=0)
    test_features_partition5 = np.concatenate((test_left_features_partition5, test_right_features_partition5), axis=0)
    # print("features_partition1.shape:", features_partition1.shape)
    # print("features_partition2.shape:", features_partition2.shape)
    # print("features_partition3.shape:", features_partition3.shape)
    # print("features_partition4.shape:", features_partition4.shape)
    # print("features_partition5.shape:", features_partition5.shape)

    labels = np.concatenate((left_labels, right_labels), axis=0)
    test_labels = np.concatenate((test_left_labels, test_right_labels), axis=0)
    # print("labels.shape:", labels.shape)

    # 把左手和右手拼接起来
    # features = np.concatenate((left_features, right_features), axis=0)
    # labels = np.concatenate((left_labels, right_labels), axis=0)
    # print(features.shape)  # (10400, 64, 5, 3) float64
    # print(labels.shape)  # (10400) float64

    # # --------------------------------------------------------------------------

    batch_size = 128

    # 定义H-CNN 输入 [batch_size, in_channel, 5, 3]
    class H_CNN(nn.Module):
        def __init__(self, in_channel, out_channel, stride=1, padding=0):
            super(H_CNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(5, 1))
            self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 3))
            self.bn = nn.BatchNorm2d(out_channel)

        def forward(self, x):
            # print("x.shape:", x.shape)
            x1 = self.conv1(x)
            # print("x1.shape:", x1.shape)
            x2 = self.conv2(x)
            # print("x2.shape:", x2.shape)
            x = x2 * x1
            x = F.relu(self.bn(x))
            return x    # 输出也得是[batch_size, out_channel, 5, 3]

    # 定义H-RNN 输入 torch.Size([32, 3, 10*5])  (batch_size, sequence_length, hidden_size) 3:sequence 32:batch 10:channel 5：band
    class H_RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=1):
            super(H_RNN, self).__init__()
            # 定义循环神经网络
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        def forward(self, x):
            # print("x.shape前:", x.shape)
            x = self.rnn(x)
            # print("x.shape后:", x.shape)
            # print("这里1")
            return x  # 输出 torch.Size([32, 3, 100])  (batch_size, sequence_length, hidden_size)

    # 定义空间注意力 Input shape: torch.Size([32, 64, 5, 3])
    class SpatialAttention(nn.Module):
        def __init__(self, in_channels):
            super(SpatialAttention, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)  # 这个卷积层用于生成用于空间注意力的权重
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # print("x的形状：", x.shape)
            # 全局平均池化
            avg_pool = F.avg_pool2d(x, x.size()[2:])
            # 卷积层和sigmoid激活函数
            weight = self.sigmoid(self.conv1(avg_pool))
            # 权重矩阵和输入特征相乘
            x = x * weight
            # print("xde的形状：", x.shape)
            return x  # Output shape after spatial attention: torch.Size([32, 64, 5, 3])

    # 定义时间注意力 Input shape: (batch_size, time_steps, feature_dim) torch.Size([32, 3, 128])
    class TimeAttention(nn.Module):
        def __init__(self, input_size, hidden_size):
            # input_size = feature_dim hidden_size = 64
            super(TimeAttention, self).__init__()
            # 定义线性层，用于将输入映射到查询空间
            self.W_q = nn.Linear(input_size, hidden_size)
            # 定义线性层，用于将输入映射到键空间
            self.W_k = nn.Linear(input_size, hidden_size)
            # 定义线性层，用于计算注意力得分
            self.V = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # 将输入投影到查询和键空间
            q = self.W_q(x)
            k = self.W_k(x)
            # 计算注意力得分
            attn_scores = self.V(torch.tanh(q + k))
            # 对得分应用softmax，得到注意力权重
            attn_weights = F.softmax(attn_scores, dim=1)
            # 输入特征加权求和
            # output = torch.sum(attn_weights * x, dim=1, keepdim=True)  # dim=1 意味着在序列长度的维度上求和，结果是一个维度为 (batch_size, 1, input_size) 的张量
            output = attn_weights * x
            return output  # Output shape after spatial attention: torch.Size([32, 1, 128])

    # 定义MLP分类器
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            # 定义网络的层
            self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层的全连接层
            self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层的全连接层
            self.relu = nn.ReLU()  # 激活函数（这里使用ReLU）
            self.dropout = nn.Dropout(0.3)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            # 前向传播过程
            x = self.fc1(x)  # 输入经过第一个全连接层
            x = self.dropout(x)
            x = self.relu(x)  # 应用激活函数
            x = self.fc2(x)  # 隐藏层输出到输出层
            x = self.dropout(x)
            # x = torch.sigmoid(x)
            x = self.softmax(x)
            return x

    # 整合模型
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            # print("进入模型")
            # 脑区1
            self.cnn1 = H_CNN(8, 64)
            self.rnn1 = H_RNN(40, 40)
            self.spatial_attention1 = SpatialAttention(64)
            self.temporal_attention1 = TimeAttention(80, 64)
            self.mlp1 = MLP(1200, 600, 2)
            # 脑区2
            self.cnn2 = H_CNN(18, 64)
            self.rnn2 = H_RNN(90, 90)
            self.spatial_attention2 = SpatialAttention(64)
            self.temporal_attention2 = TimeAttention(180, 64)
            self.mlp2 = MLP(1500, 750, 2)
            # 脑区3
            self.cnn3 = H_CNN(9, 64)
            self.rnn3 = H_RNN(45, 45)
            self.spatial_attention3 = SpatialAttention(64)
            self.temporal_attention3 = TimeAttention(90, 64)
            self.mlp3 = MLP(1230, 615, 2)
            # 脑区4
            self.cnn4 = H_CNN(20, 64)
            self.rnn4 = H_RNN(100, 100)
            self.spatial_attention4 = SpatialAttention(64)
            self.temporal_attention4 = TimeAttention(200, 64)
            self.mlp4 = MLP(1560, 780, 2)
            # 脑区5
            self.cnn5 = H_CNN(9, 64)
            self.rnn5 = H_RNN(45, 45)
            self.spatial_attention5 = SpatialAttention(64)
            self.temporal_attention5 = TimeAttention(90, 64)
            self.mlp5 = MLP(1230, 615, 2)

            self.mlp = MLP(6720, 3360, 2)

        def forward(self, x1, x2, x3, x4, x5):
            # 脑区1
            x_cnn1 = x1
            cnn_features1 = self.cnn1(x_cnn1)
            # x_rnn1 = np.transpose(x1, (0, 3, 1, 2))
            x_rnn1 = x1.permute(0, 3, 1, 2)
            x_rnn1 = x_rnn1.reshape(batch_size, 3, 40)
            rnn_features1, _ = self.rnn1(x_rnn1)
            spatial_features1 = self.spatial_attention1(cnn_features1)
            temporal_features1 = self.temporal_attention1(rnn_features1)
            spatial_features1 = spatial_features1.reshape(batch_size, -1)
            temporal_features1 = temporal_features1.reshape(batch_size, -1)
            merged_features1 = torch.concatenate((spatial_features1, temporal_features1), dim=1)
            # print("merged_features1.shape:", merged_features1.shape)

            # 脑区2
            x_cnn2 = x2
            cnn_features2 = self.cnn2(x_cnn2)
            # x_rnn2 = np.transpose(x2, (0, 3, 1, 2))
            x_rnn2 = x2.permute(0, 3, 1, 2)
            x_rnn2 = x_rnn2.reshape(batch_size, 3, 90)
            rnn_features2, _ = self.rnn2(x_rnn2)
            spatial_features2 = self.spatial_attention2(cnn_features2)
            temporal_features2 = self.temporal_attention2(rnn_features2)
            spatial_features2 = spatial_features2.reshape(batch_size, -1)
            temporal_features2 = temporal_features2.reshape(batch_size, -1)
            merged_features2 = torch.concatenate((spatial_features2, temporal_features2), dim=1)
            # print("merged_features2.shape:", merged_features2.shape)

            # 脑区3
            x_cnn3 = x3
            cnn_features3 = self.cnn3(x_cnn3)
            # x_rnn3 = np.transpose(x3, (0, 3, 1, 2))
            x_rnn3 = x3.permute(0, 3, 1, 2)
            x_rnn3 = x_rnn3.reshape(batch_size, 3, 45)
            rnn_features3, _ = self.rnn3(x_rnn3)
            spatial_features3 = self.spatial_attention3(cnn_features3)
            temporal_features3 = self.temporal_attention3(rnn_features3)
            spatial_features3 = spatial_features3.reshape(batch_size, -1)
            temporal_features3 = temporal_features3.reshape(batch_size, -1)
            merged_features3 = torch.concatenate((spatial_features3, temporal_features3), dim=1)
            # print("merged_features3.shape:", merged_features3.shape)

            # 脑区4
            x_cnn4 = x4
            cnn_features4 = self.cnn4(x_cnn4)
            # x_rnn4 = np.transpose(x4, (0, 3, 1, 2))
            x_rnn4 = x4.permute(0, 3, 1, 2)
            x_rnn4 = x_rnn4.reshape(batch_size, 3, 100)
            rnn_features4, _ = self.rnn4(x_rnn4)
            spatial_features4 = self.spatial_attention4(cnn_features4)
            temporal_features4 = self.temporal_attention4(rnn_features4)
            spatial_features4 = spatial_features4.reshape(batch_size, -1)
            temporal_features4 = temporal_features4.reshape(batch_size, -1)
            merged_features4 = torch.concatenate((spatial_features4, temporal_features4), dim=1)
            # print("merged_features4.shape:", merged_features4.shape)

            # 脑区5
            x_cnn5 = x5
            cnn_features5 = self.cnn5(x_cnn5)
            # x_rnn5 = np.transpose(x5, (0, 3, 1, 2))
            x_rnn5 = x5.permute(0, 3, 1, 2)
            x_rnn5 = x_rnn5.reshape(batch_size, 3, 45)
            rnn_features5, _ = self.rnn5(x_rnn5)
            spatial_features5 = self.spatial_attention5(cnn_features5)
            temporal_features5 = self.temporal_attention5(rnn_features5)
            spatial_features5 = spatial_features5.reshape(batch_size, -1)
            temporal_features5 = temporal_features5.reshape(batch_size, -1)
            merged_features5 = torch.concatenate((spatial_features5, temporal_features5), dim=1)
            # print("merged_features5.shape:", merged_features5.shape)

            merged_features = torch.cat((merged_features1, merged_features2, merged_features3, merged_features4, merged_features5), dim=1)
            # print("total_merged_features.shape:", merged_features.shape)

            # MLP分类
            output = self.mlp(merged_features)
            return output

    # --------------------------------------------------------------------------
    # 将数组转换为张量
    features_partition1 = torch.tensor(features_partition1, dtype=torch.float32)
    features_partition2 = torch.tensor(features_partition2, dtype=torch.float32)
    features_partition3 = torch.tensor(features_partition3, dtype=torch.float32)
    features_partition4 = torch.tensor(features_partition4, dtype=torch.float32)
    features_partition5 = torch.tensor(features_partition5, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    # 测试集
    test_features_partition1 = torch.tensor(test_features_partition1, dtype=torch.float32)
    test_features_partition2 = torch.tensor(test_features_partition2, dtype=torch.float32)
    test_features_partition3 = torch.tensor(test_features_partition3, dtype=torch.float32)
    test_features_partition4 = torch.tensor(test_features_partition4, dtype=torch.float32)
    test_features_partition5 = torch.tensor(test_features_partition5, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    dataset = TensorDataset(features_partition1, features_partition2, features_partition3, features_partition4, features_partition5, labels)
    # print("dataset", dataset.__len__())  #  10200

    # print("test_features_partition1.shape", test_features_partition1.shape)
    # print("test_features_partition2.shape", test_features_partition2.shape)
    # print("test_features_partition3.shape", test_features_partition3.shape)
    # print("test_features_partition4.shape", test_features_partition4.shape)
    # print("test_features_partition5.shape", test_features_partition5.shape)
    # print("test_labels.shape", test_labels.shape)
    test_dataset = TensorDataset(test_features_partition1, test_features_partition2, test_features_partition3, test_features_partition4, test_features_partition5, test_labels)
    # print("test_dataset", test_dataset.__len__())  # 200

    # 把dataset划分训练和测试
    # split_ratio = 0.9
    # total_samples = len(dataset)
    # train_size = int(split_ratio * total_samples)
    # test_size = total_samples - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset = dataset
    # test_dataset = test_dataset

    train_dataloader_partition = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    # 创建整合模型实例
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MyModel().to(device)

    # 加载模型参数
    model.load_state_dict(torch.load('D:\pythonProject\svr\CHO_model_TSNE\TSNE_model.pth'))

    # 模型训练和测试
    num_epochs = 1
    best_accuracy = 0.0  # 用于跟踪最高准确率
    print("下面是被试{}作为测试集，剩下的作为训练集的训练集和测试集准确率：".format(num))
    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0

        # 模型评估
        # print("下面是被试一的测试集准确率：")
        with torch.no_grad():
            for data in test_dataloader:  # 假设使用 DataLoader 加载测试数据
                x1, x2, x3, x4, x5, labels = data
                x1 = x1.to(device)
                x2 = x2.to(device)
                x3 = x3.to(device)
                x4 = x4.to(device)
                x5 = x5.to(device)
                # labels = labels.to(device)
                outputs = model(x1, x2, x3, x4, x5)
                # print(outputs.shape)
                # print(labels)
                # print("outputs.shape:")
                # print(labels.shape)
                outputs = outputs.cpu()
                # labels = labels.view(-1, 1)
                # labels = labels.float()
                # loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            accuracy = total_correct / total_samples
            # print("total_correct.number:")
            # print(total_correct)
            # print("total_samples.number:")
            # print(total_samples)
            # if accuracy > best_accuracy:
            #     best_accuracy = accuracy
            #     best_model_state = model.state_dict()
            # print(f'                Loss: {loss.item():.4f}, Test Accuracy: {accuracy * 100:.2f}%')
            print(f'Test Accuracy: {accuracy * 100:.2f}%')
            # print(outputs)
            # print(labels)

            # # 频带实验
            # recall = recall_score(labels, predicted, average='macro')
            # print("Recall:", recall)
            # precision = precision_score(labels, predicted, average='macro')
            # print("Precision:", precision)
            # f1 = f1_score(labels, predicted, average='macro')
            # print("F1 Score:", f1)
            # kappa = cohen_kappa_score(labels, predicted)
            # print("Cohen's Kappa:", kappa)

            # 设置全局字体大小
            plt.rcParams.update({'font.size': 15})
            tick_labels = ['LH', 'RH']

            # 计算混淆矩阵
            confusion_mat = confusion_matrix(labels, predicted)

            # 归一化混淆矩阵
            normalized_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

            # 将数值格式化为百分比
            percentage_mat = np.around(normalized_mat, decimals=2)

            # 绘制混淆矩阵图形
            plt.figure(figsize=(8, 6))
            sns.heatmap(percentage_mat, annot=True, cmap='Reds', fmt='.2f', cbar=False)

            # 设置 x 轴和 y 轴刻度标签
            # plt.xticks(np.arange(len(tick_labels)), tick_labels)
            # plt.yticks(np.arange(len(tick_labels)), tick_labels)
            plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, ha='center')
            plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, va='center')

            # 设置图形的标题、轴标签和颜色栏
            plt.title(title)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.show()

            plt.close()

            # # 实例化t-SNE模型
            # tsne = TSNE(n_components=2)
            #
            # # 对数据进行降维
            # X_embedded = tsne.fit_transform(outputs)
            #
            # # 绘制降维后的数据
            # plt.figure(figsize=(8, 6))
            # # plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
            # plt.scatter(X_embedded[labels == 0, 0], X_embedded[labels == 0, 1], c='r', marker='o', s=150, label='left hand')
            # plt.scatter(X_embedded[labels == 1, 0], X_embedded[labels == 1, 1], c='b', marker='^', s=150, label='right hand')
            # plt.title('two_class_of_S3T')
            # plt.xlabel('')
            # plt.ylabel('')
            # # 添加图例
            # plt.legend()
            # plt.show()
