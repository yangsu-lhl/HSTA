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
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

best_accuracy_list = []
for num in range(3, 4):

    # 从特征文件夹和标签文件夹加载数据
    feature_folder = "D:\postgraduate\MIdata\physionet\libo\physionet\DE/"
    label_folder = "D:\postgraduate\MIdata\physionet\libo\physionet\label/"

    # 从特征文件夹中加载数据
    features = []
    for i in range(num, num+1):
        feature_file = np.load(feature_folder + "S{}.npy".format(i))
        features.append(feature_file)

    # 从标签文件夹中加载数据
    labels = []
    for i in range(num, num+1):
        label_file = np.load(label_folder + "S{}.npy".format(i))
        labels.append(label_file)

    features = np.array(features)
    labels = np.array(labels)
    # print(features.shape)  # (1, 360, 64, 5)
    # print(labels.shape)  # (1, 360, 1)
    # print(labels)
    features = features.reshape(-1, 90, 4, 64, 5)
    features = np.transpose(features, (0, 1, 3, 4, 2))
    features = features.reshape((-1, 64, 5, 4))
    # print(features.shape)  # (90, 64, 5, 4)
    labels = labels.reshape((-1, 90, 4, 1)).mean(axis=2)
    labels = labels.reshape(-1)
    # print(labels)
    # print(labels.shape)  # (90,)

    # 接下来要把64个通道分为5个区域 基于国际10-10系统的64通道蒙太奇脑电图技术
    # 定义5个区域
    features_partition1 = []  # 1-3,33-37
    features_partition2 = []  # 4-11,38-47
    features_partition3 = []  # 12-15,48-52
    features_partition4 = []  # 16-24,31-32,53-61
    features_partition5 = []  # 25-30,62-64

    features_partition1_1 = features[:, 0:3, :, :]
    features_partition1_2 = features[:, 32:37, :, :]
    features_partition1 = np.concatenate((features_partition1_1, features_partition1_2), axis=1)
    # print(features_partition1.shape)  # (8100, 8, 5, 4)

    features_partition2_1 = features[:, 3:11, :, :]
    features_partition2_2 = features[:, 37:47, :, :]
    features_partition2 = np.concatenate((features_partition2_1, features_partition2_2), axis=1)
    # print(features_partition2.shape)  # (8100, 18, 5, 4)

    features_partition3_1 = features[:, 11:15, :, :]
    features_partition3_2 = features[:, 47:52, :, :]
    features_partition3 = np.concatenate((features_partition3_1, features_partition3_2), axis=1)
    # print(features_partition3.shape)  # (8100, 9, 5, 4)

    features_partition4_1 = features[:, 15:24, :, :]
    features_partition4_2 = features[:, 30:32, :, :]
    features_partition4_3 = features[:, 52:61, :, :]
    features_partition4 = np.concatenate((features_partition4_1, features_partition4_2, features_partition4_3), axis=1)
    # print(features_partition4.shape)  # (8100, 20, 5, 4)

    features_partition5_1 = features[:, 24:30, :, :]
    features_partition5_2 = features[:, 61:64, :, :]
    features_partition5 = np.concatenate((features_partition5_1, features_partition5_2), axis=1)
    # print(features_partition5.shape)  # (8100, 9, 5, 4)

    # print(labels.shape)  # (8100,)

    batch_size = 20

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
            # 卷积层和sigmoid激活函数
            weight = self.sigmoid(self.conv1(avg_pool))
            # 权重矩阵和输入特征相乘
            x = x * weight
            # print("xde的形状：", x.shape)
            return x  # Output shape after spatial attention: torch.Size([32, 64, 5, 4])

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
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            # 定义网络的层
            self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层的全连接层
            self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层的全连接层
            self.relu = nn.ReLU()  # 激活函数（这里使用ReLU）
            self.dropout = nn.Dropout(0.2)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            # 前向传播过程
            x = self.fc1(x)  # 输入经过第一个全连接层
            x = self.dropout(x)
            x = self.relu(x)  # 应用激活函数
            x = self.fc2(x)  # 隐藏层输出到输出层
            # x = self.dropout(x)
            x = self.softmax(x)  # 使用Softmax输出概率分布
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
            self.mlp1 = MLP(1600, 800, 4)
            # 脑区2
            self.cnn2 = H_CNN(18, 64)
            self.rnn2 = H_RNN(90, 90)
            self.spatial_attention2 = SpatialAttention(64)
            self.temporal_attention2 = TimeAttention(180, 64)
            self.mlp2 = MLP(2000, 1000, 4)
            # 脑区3
            self.cnn3 = H_CNN(9, 64)
            self.rnn3 = H_RNN(45, 45)
            self.spatial_attention3 = SpatialAttention(64)
            self.temporal_attention3 = TimeAttention(90, 64)
            self.mlp3 = MLP(1640, 820, 4)
            # 脑区4
            self.cnn4 = H_CNN(20, 64)
            self.rnn4 = H_RNN(100, 100)
            self.spatial_attention4 = SpatialAttention(64)
            self.temporal_attention4 = TimeAttention(200, 64)
            self.mlp4 = MLP(2080, 1040, 4)
            # 脑区5
            self.cnn5 = H_CNN(9, 64)
            self.rnn5 = H_RNN(45, 45)
            self.spatial_attention5 = SpatialAttention(64)
            self.temporal_attention5 = TimeAttention(90, 64)
            self.mlp5 = MLP(1640, 820, 4)

            self.mlp = MLP(8960, 4480, 4)

        def forward(self, x1, x2, x3, x4, x5):
            # 脑区1
            x_cnn1 = x1
            # print("x_cnn1.shape:", x_cnn1.shape)torch.Size([20, 8, 5, 4])
            cnn_features1 = self.cnn1(x_cnn1)
            # print("cnn_features1.shape:", cnn_features1.shape)torch.Size([20, 64, 5, 4])
            # x_rnn1 = np.transpose(x1, (0, 3, 1, 2))
            x_rnn1 = x1.permute(0, 3, 1, 2)
            # print("x_rnn1.shape:", x_rnn1.shape)torch.Size([20, 4, 8, 5])
            x_rnn1 = x_rnn1.reshape(batch_size, 4, 40)
            # print("x_rnn1.shape:", x_rnn1.shape)torch.Size([20, 4, 40])
            rnn_features1, _ = self.rnn1(x_rnn1)
            # print("rnn_features1.shape:", rnn_features1.shape)torch.Size([20, 4, 80])
            spatial_features1 = self.spatial_attention1(cnn_features1)
            temporal_features1 = self.temporal_attention1(rnn_features1)
            # print("spatial_features1.shape:", spatial_features1.shape)torch.Size([20, 64, 5, 4])
            # print("temporal_features1.shape:", temporal_features1.shape)torch.Size([20, 4, 80])
            spatial_features1 = spatial_features1.reshape(batch_size, -1)
            temporal_features1 = temporal_features1.reshape(batch_size, -1)
            # print("spatial_features1.shape:", spatial_features1.shape)torch.Size([20, 1280])
            # print("temporal_features1.shape:", temporal_features1.shape)torch.Size([20, 320])
            merged_features1 = torch.concatenate((spatial_features1, temporal_features1), dim=1)
            # print("merged_features1.shape:", merged_features1.shape)torch.Size([20, 1600])

            # 脑区2
            x_cnn2 = x2
            cnn_features2 = self.cnn2(x_cnn2)
            # x_rnn2 = np.transpose(x2, (0, 3, 1, 2))
            x_rnn2 = x2.permute(0, 3, 1, 2)
            x_rnn2 = x_rnn2.reshape(batch_size, 4, 90)
            rnn_features2, _ = self.rnn2(x_rnn2)
            # print("脑区二")
            # print("cnn_features2.shape:", cnn_features2.shape)
            # print("rnn_features2.shape:", rnn_features2.shape)
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
            x_rnn3 = x_rnn3.reshape(batch_size, 4, 45)
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
            x_rnn4 = x_rnn4.reshape(batch_size, 4, 100)
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
            x_rnn5 = x_rnn5.reshape(batch_size, 4, 45)
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

    # 将数组转换为张量
    features_partition1 = torch.tensor(features_partition1, dtype=torch.float32)
    features_partition2 = torch.tensor(features_partition2, dtype=torch.float32)
    features_partition3 = torch.tensor(features_partition3, dtype=torch.float32)
    features_partition4 = torch.tensor(features_partition4, dtype=torch.float32)
    features_partition5 = torch.tensor(features_partition5, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    print(features_partition1.shape)
    print(labels.shape)

    dataset = TensorDataset(features_partition1, features_partition2, features_partition3, features_partition4, features_partition5, labels)

    # 把dataset划分训练和测试
    split_ratio = 0.7
    total_samples = len(dataset)
    train_size = int(split_ratio * total_samples)
    test_size = total_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    # 创建整合模型实例
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MyModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    # 模型训练和测试
    num_epochs = 30
    best_accuracy = 0.0  # 用于跟踪最高准确率
    best_model_state = None  # 用于保存最佳模型状态
    # Train_loss_list = []
    # Train_accuracy_list = []
    # Test_loss_list = []
    # Test_accuracy_list = []
    print("下面是被试{}的训练集和测试集准确率：".format(num))
    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0
        for data in train_dataloader:  # 使用 DataLoader 加载数据8
            x1, x2, x3, x4, x5, y = data
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            x5 = x5.to(device)
            y = y.to(device)
            outputs = model(x1, x2, x3, x4, x5)
            # y = y.view(-1, 1)
            # y = y.float()
            # print(outputs)
            print(555)
            print(outputs.shape)  # torch.Size([20, 4])
            print(outputs)
            # print(y)
            print(y.shape)  # torch.Size([20])
            print(y)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 在多分类任务中，通常使用argmax来确定模型预测的类别
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == y).sum().item()
            total_samples += y.size(0)
            # print(total_correct)
            # print(total_samples)
            # predicted = (outputs >= 0.5).float()  # 将张量中大于等于 0.5 的值设为 1，小于 0.5 的值设为 0
            # total_correct += (predicted == y).sum().item()
            # total_samples += y.size(0)
        accuracy = total_correct / total_samples
        # Train_loss_list.append(loss.item())
        # Train_accuracy_list.append(accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {accuracy * 100:.2f}%')

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
                labels = labels.to(device)
                outputs = model(x1, x2, x3, x4, x5)
                # labels = labels.view(-1, 1)
                # labels = labels.float()
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                # predicted = (outputs >= 0.5).float()  # 将张量中大于等于 0.5 的值设为 1，小于 0.5 的值设为 0
                # total_correct += (predicted == labels).sum().item()
                # total_samples += labels.size(0)
            accuracy = total_correct / total_samples
            # Test_loss_list.append(loss.item())
            # Test_accuracy_list.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict()
        print(f'                Loss: {loss.item():.4f}, Test Accuracy: {accuracy * 100:.2f}%')

    print(f'测试集准确率最高为：{best_accuracy * 100:.2f}%')
    best_accuracy_list.append(best_accuracy * 100)

    folder_path = "D:\pythonProject\svr/physionet_model"
    file_name = "subject_{}_best_model.pth".format(num)
    file_path = os.path.join(folder_path, file_name)
    # torch.save(best_model_state, file_path)

print("测试集最高准确率列表:")
print(best_accuracy_list)
# print("测试集损失列表:")
# print(Test_loss_list)
# print("测试集准确率列表:")
# print(Test_accuracy_list)

# 创建一个DataFrame对象
df = pd.DataFrame(best_accuracy_list, columns=["test_best_accuracy"])
# 保存DataFrame为Excel文件
file_path = "physionet_test_best_accuracy.xlsx"  # 文件路径和名称
# df.to_excel(file_path, index=False)  # 将数据保存到Excel文件中，不包含索引列

# 计算总和
total = sum(best_accuracy_list)
# 计算平均值
average = total / len(best_accuracy_list)
# 打印平均值
print("测试集平均准确率为:", average)

print("所有被试全部跑完！！！！！！！！！！！")

# # 画 训练集loss 曲线
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)  # 创建一个 1x2 的子图，使用第一个子图
# plt.plot(Train_loss_list, label='Train Loss')
# plt.title('Loss Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# # 画 训练集accuracy 曲线
# plt.subplot(1, 2, 2)  # 使用第二个子图
# plt.plot(Train_accuracy_list, label='Train Accuracy', color='green')
# plt.title('Accuracy Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# # 展示图形
# plt.tight_layout()  # 确保子图之间的布局良好
# plt.show()
#
# # 画 测试集loss 曲线
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)  # 创建一个 1x2 的子图，使用第一个子图
# plt.plot(Test_loss_list, label='Test Loss')
# plt.title('Loss Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# # 画 测试集accuracy 曲线
# plt.subplot(1, 2, 2)  # 使用第二个子图
# plt.plot(Test_accuracy_list, label='Test Accuracy', color='green')
# plt.title('Accuracy Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# # 展示图形
# plt.tight_layout()  # 确保子图之间的布局良好
# plt.show()
