import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

batch_size = 90

# 定义H-CNN 输入 [batch_size, in_channel, 5, 4]
class H_CNN(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, padding=0):
        super(H_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(5, 1))
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 4))
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        # print("x.shape:", x.shape)
        x1 = self.conv1(x)
        # print("x1.shape:", x1.shape)
        x2 = self.conv2(x)
        # print("x2.shape:", x2.shape)
        x = x2 * x1
        x = F.relu(self.bn(x))
        return x    # 输出也得是[batch_size, out_channel, 5, 4]

# 定义H-RNN 输入 torch.Size([32, 4, 10*5])  (batch_size, sequence_length, hidden_size) 4:sequence 32:batch 10:channel 5：band
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
        return x  # 输出 torch.Size([32, 4, 100])  (batch_size, sequence_length, hidden_size)

# 定义空间注意力 Input shape: torch.Size([32, 64, 5, 4])
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)  # 这个卷积层用于生成用于空间注意力的权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("x的形状：", x.shape)
        # 全局平均池化
        avg_pool = F.avg_pool2d(x, x.size()[2:])
        # print(avg_pool.shape)
        # 卷积层和sigmoid激活函数
        weight = self.sigmoid(self.conv1(avg_pool))
        # 权重矩阵和输入特征相乘
        x = x * weight
        # print("xde的形状：", x.shape)
        return x
        # Output shape after spatial attention: torch.Size([32, 64, 5, 4])

# 定义时间注意力 Input shape: (batch_size, time_steps, feature_dim) torch.Size([32, 4, 128])
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
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        # 定义网络的层
        self.fc1 = nn.Linear(input_size, hidden_size1)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # 隐藏层到输出层的全连接层
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()  # 激活函数（这里使用ReLU）
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 前向传播过程
        x = self.fc1(x)  # 输入经过第一个全连接层
        x = self.dropout(x)
        x = self.relu(x)  # 应用激活函数
        x2 = self.fc2(x)  # 隐藏层输出到输出层
        x = self.fc3(x2)
        x = self.softmax(x)  # 使用Softmax输出概率分布
        return x, x2

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
        self.mlp1 = MLP(1600, 800, 64, 4)
        # 脑区2
        self.cnn2 = H_CNN(18, 64)
        self.rnn2 = H_RNN(90, 90)
        self.spatial_attention2 = SpatialAttention(64)
        self.temporal_attention2 = TimeAttention(180, 64)
        self.mlp2 = MLP(2000, 1000, 64, 4)
        # 脑区3
        self.cnn3 = H_CNN(9, 64)
        self.rnn3 = H_RNN(45, 45)
        self.spatial_attention3 = SpatialAttention(64)
        self.temporal_attention3 = TimeAttention(90, 64)
        self.mlp3 = MLP(1640, 820, 64, 4)
        # 脑区4
        self.cnn4 = H_CNN(20, 64)
        self.rnn4 = H_RNN(100, 100)
        self.spatial_attention4 = SpatialAttention(64)
        self.temporal_attention4 = TimeAttention(200, 64)
        self.mlp4 = MLP(2080, 1040, 64, 4)
        # 脑区5
        self.cnn5 = H_CNN(9, 64)
        self.rnn5 = H_RNN(45, 45)
        self.spatial_attention5 = SpatialAttention(64)
        self.temporal_attention5 = TimeAttention(90, 64)
        self.mlp5 = MLP(1640, 820, 64, 4)

        self.mlp = MLP(8960, 4480, 64, 4)
        # self.mlp = MLP(1640, 820, 64, 4)

    def forward(self, x1, x2, x3, x4, x5):
    # def forward(self, x3):
        # # 脑区1
        x_cnn1 = x1
        cnn_features1 = self.cnn1(x_cnn1)
        # x_rnn1 = np.transpose(x1, (0, 3, 1, 2))
        x_rnn1 = x1.permute(0, 3, 1, 2)
        x_rnn1 = x_rnn1.reshape(batch_size, 4, 40)
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
        x_rnn2 = x_rnn2.reshape(batch_size, 4, 90)
        rnn_features2, _ = self.rnn2(x_rnn2)
        spatial_features2 = self.spatial_attention2(cnn_features2)
        temporal_features2 = self.temporal_attention2(rnn_features2)
        spatial_features2 = spatial_features2.reshape(batch_size, -1)
        temporal_features2 = temporal_features2.reshape(batch_size, -1)
        merged_features2 = torch.concatenate((spatial_features2, temporal_features2), dim=1)
        # # print("merged_features2.shape:", merged_features2.shape)
        #
        # # 脑区3
        x_cnn3 = x3
        # print("x_cnn3:", x_cnn3.shape)  # x_cnn3: torch.Size([64, 9, 5, 4])
        cnn_features3 = self.cnn3(x_cnn3)
        # print("cnn_features3:", cnn_features3.shape)  # cnn_features3: torch.Size([64, 64, 5, 4])
        # x_rnn3 = np.transpose(x3, (0, 3, 1, 2))
        x_rnn3 = x3.permute(0, 3, 1, 2)
        x_rnn3 = x_rnn3.reshape(batch_size, 4, 45)
        rnn_features3, _ = self.rnn3(x_rnn3)
        spatial_features3 = self.spatial_attention3(cnn_features3)
        temporal_features3 = self.temporal_attention3(rnn_features3)
        spatial_features3 = spatial_features3.reshape(batch_size, -1)
        temporal_features3 = temporal_features3.reshape(batch_size, -1)
        merged_features3 = torch.concatenate((spatial_features3, temporal_features3), dim=1)
        # # print("merged_features3.shape:", merged_features3.shape)
        #
        # # 脑区4
        x_cnn4 = x4
        cnn_features4 = self.cnn4(x_cnn4)
        # x_rnn4 = np.transpose(x4, (0, 3, 1, 2))
        x_rnn4 = x4.permute(0, 3, 1, 2)
        x_rnn4 = x_rnn4.reshape(batch_size, 4, 100)
        rnn_features4, _ = self.rnn4(x_rnn4)
        spatial_features4 = self.spatial_attention4(cnn_features4)
        temporal_features4 = self.temporal_attention4(rnn_features4)
        spatial_features4 = spatial_features4.reshape(batch_size, -1)
        temporal_features4 = temporal_features4.reshape(batch_size, -1)
        merged_features4 = torch.concatenate((spatial_features4, temporal_features4), dim=1)
        # # print("merged_features4.shape:", merged_features4.shape)
        #
        # # 脑区5
        x_cnn5 = x5
        cnn_features5 = self.cnn5(x_cnn5)
        # x_rnn5 = np.transpose(x5, (0, 3, 1, 2))
        x_rnn5 = x5.permute(0, 3, 1, 2)
        x_rnn5 = x_rnn5.reshape(batch_size, 4, 45)
        rnn_features5, _ = self.rnn5(x_rnn5)
        spatial_features5 = self.spatial_attention5(cnn_features5)
        temporal_features5 = self.temporal_attention5(rnn_features5)
        spatial_features5 = spatial_features5.reshape(batch_size, -1)
        temporal_features5 = temporal_features5.reshape(batch_size, -1)
        merged_features5 = torch.concatenate((spatial_features5, temporal_features5), dim=1)
        # # print("merged_features5.shape:", merged_features5.shape)

        merged_features = torch.cat((merged_features1, merged_features2, merged_features3, merged_features4, merged_features5), dim=1)
        # merged_features = merged_features3
        # print("total_merged_features.shape:", merged_features.shape)

        # MLP分类
        output, brain = self.mlp(merged_features)
        return output, brain