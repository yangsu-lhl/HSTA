import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

# 从特征文件夹和标签文件夹加载数据
feature_folder = "D:\postgraduate\MIdata/52_subject/feature_DE1/"
label_folder = "D:\postgraduate\MIdata/52_subject\label/"

# 从特征文件夹中加载数据
features = []
for i in range(1, 53):
    feature_file = np.load(feature_folder + "feature_DE{:02}.npy".format(i))
    if i == 7 or i == 9 or i == 46:
        feature_file = feature_file[:600, :, :]
    # print(feature_file.shape)
    features.append(feature_file)

# 从标签文件夹中加载数据
labels = []
for i in range(1, 53):
    label_file = np.load(label_folder + "label{:02}.npy".format(i))
    if i == 7 or i == 9 or i == 46:
        label_file = label_file[:600]
    # print(label_file.shape)
    labels.append(label_file)

features = np.array(features)
labels = np.array(labels)
features = features.reshape(-1, 64, 5)  # (31200, 64, 5)
labels = labels.reshape(-1)  # (31200,)

# 把数据集放到dataloader中
batch_size = 64
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = MyDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

split_ratio = 0.8
total_samples = len(dataset)
train_size = int(split_ratio * total_samples)
test_size = total_samples - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 验证模型
class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(64)
    def forward(self, x):
        # print(x.shape)  # (batch_size, channel, frequency)  torch.Size([32, 64, 5])
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x_ = torch.matmul(x2, x1)  # (B,C1,C2)
        x__ = x * x_
        # print(x1.shape, x2.shape, x_.shape, x__.shape)
        # torch.Size([32, 1, 5]) torch.Size([32, 64, 1]) torch.Size([32, 64, 5]) torch.Size([32, 64, 5])
        return x__

class conv_model(nn.Module):
    def __init__(self):
        super(conv_model, self).__init__()
        self.conv1 = nn.Conv1d(64, 64, kernel_size=1)
        self.attention = attention()
        self.fc1 = nn.Linear(64*5, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):

        x_ = F.relu(self.conv1(x))
        # print(x_.shape)  # torch.Size([32, 64, 5])
        attention_ = self.attention(x_)
        # print(attention_.shape)  # torch.Size([32, 64, 5])
        fusion = attention_.view(-1, 64*5)  # (32, 320)
        out1 = self.fc1(fusion)
        out1 = F.relu(out1)
        out2 = self.fc2(out1)
        out2 = F.relu(out2)
        out3 = self.fc3(out2)
        out_ = torch.sigmoid(out3)
        print("out shape:", out_.shape)
        return out_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

myModel = conv_model().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.0001, weight_decay=0.001)

# 模型训练
num_epochs = 20
for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0
    for data in train_dataloader:  # 使用 DataLoader 加载数据
        x, y = data
        x = x.to(device)
        y = y.to(device)
        outputs = myModel(x)
        y = y.view(-1, 1)
        y = y.float()
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # _, predicted = torch.max(outputs, 1)
        predicted = (outputs >= 0.5).float()  # 将张量中大于等于 0.5 的值设为 1，小于 0.5 的值设为 0
        total_correct += (predicted == y).sum().item()
        total_samples += y.size(0)
    accuracy = total_correct / total_samples
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%')

# # 模型评估
# myModel.eval()
# with torch.no_grad():
#     for inputs, labels in test_dataloader:  # 假设使用 DataLoader 加载测试数据
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         outputs = myModel(inputs)
#         _, predicted = torch.max(outputs, 1)
#         accuracy = (predicted == labels).float().mean()
#
# print(f'Test Accuracy: {accuracy.item()}')



