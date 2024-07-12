import matplotlib.pyplot as plt
from pylab import *
from utils import *
import seaborn as sns
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn, optim
import warnings
from SEED_IV.Index_calculation import *
from SEED_IV.mix_train_test_loader import *

class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=62,out_channels=1,kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=62,out_channels=62,kernel_size=5)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(62)
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        # print(x1.shape, x2.shape)
        x_ = torch.matmul(x2, x1) #(B,C1,C2)
        x = x * x_
        return x

class conv_model(nn.Module):
    def __init__(self):
        super(conv_model, self).__init__()
        self.conv1 = nn.Conv1d(62,62,kernel_size=1)
        self.attention = attention()
        self.fc = nn.Linear(62*5, 4)
    def forward(self, x):
        # prefrontal = F.relu(nn.Conv1d(prefrontal.shape[1], prefrontal.shape[1], kernel_size=1).to(device)(prefrontal))
        # cesial_frontal = F.relu(nn.Conv1d(cesial_frontal.shape[1], cesial_frontal.shape[1], kernel_size=1).to(device)(cesial_frontal))
        # central = F.relu(nn.Conv1d(central.shape[1], central.shape[1], kernel_size=1).to(device)(central))
        # parietal = F.relu(nn.Conv1d(parietal.shape[1], parietal.shape[1], kernel_size=1).to(device)(parietal))
        # occipital = F.relu(nn.Conv1d(occipital.shape[1], occipital.shape[1], kernel_size=1).to(device)(occipital))
        # central_line = F.relu(nn.Conv1d(central_line.shape[1], central_line.shape[1], kernel_size=1).to(device)(central_line))
        # left = F.relu(nn.Conv1d(left.shape[1], left.shape[1], kernel_size=1).to(device)(left))
        # left_central = F.relu(nn.Conv1d(left_central.shape[1], left_central.shape[1], kernel_size=1).to(device)(left_central))
        # right_central = F.relu(nn.Conv1d(right_central.shape[1], right_central.shape[1], kernel_size=1).to(device)(right_central))
        # right = F.relu(nn.Conv1d(right.shape[1], right.shape[1], kernel_size=1).to(device)(right))
        x = F.relu(self.conv1(x))

        attention = self.attention(x)

        fusion = attention.view(-1,62*5)
        out = F.relu(self.fc(fusion))

        return out



warnings.filterwarnings("ignore")

learning_rate = 0.01
epochs = 200
min_acc = 0.2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

myModel = conv_model().to(device)
loss_func = nn.MSELoss()
loss_feature = nn.NLLLoss()
loss_cross = nn.CrossEntropyLoss()
# loss_common = common_loss()
opt = torch.optim.Adam(myModel.parameters(), lr=learning_rate, weight_decay=0.01)
# opt = optim.SGD(myModel.parameters(), lr=0.01, momentum=0.5)

G = testclass()
train_len = G.len(X_train.shape[0], batch_size)
test_len = G.len(X_test.shape[0], batch_size)

train_loss_plt = []
train_acc_plt = []
test_loss_plt = []
test_acc_plt = []
Train_Loss_list = []
Train_Accuracy_list = []
Test_Loss_list = []
Test_Accuracy_list = []

for i in range(epochs):
    total_train_step = 0
    total_test_step = 0

    total_train_loss = 0
    total_train_acc = 0

    for data in train_dataloader:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        output = myModel(x)
        # print(output, y)
        train_loss_task = loss_cross(output, y)

        opt.zero_grad()
        train_loss_task.backward()
        opt.step()

        train_acc = (output.argmax(dim=1)  == y).sum()
        # train_acc = G.acc(output5, y)
        # print(train_acc)

        train_loss_plt.append(train_loss_task)
        total_train_loss = total_train_loss + train_loss_task.item()
        total_train_step = total_train_step + 1

        train_acc_plt.append(train_acc)
        total_train_acc += train_acc

    Train_Loss_list.append(total_train_loss / (len(train_dataloader)))
    Train_Accuracy_list.append(total_train_acc / train_len)

    total_test_loss = 0
    total_test_acc = 0
    matrix = np.full((5,5),0)
    with torch.no_grad():
        pred_output_list = []

        for data in test_dataloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = myModel(x)
            # print(output, y)
            test_loss_task = loss_cross(output, y)
            test_acc = (output.argmax(dim=1) == y).sum()
            TP_TN_FP_FN = G.Compute_TP_TN_FP_FN_5(output.argmax(dim=1), y, matrix)

            test_loss_plt.append(test_loss_task)
            total_test_loss = total_test_loss + test_loss_task.item()
            total_test_step = total_test_step + 1

            test_acc_plt.append(test_acc)
            total_test_acc += test_acc



    Test_Loss_list.append(total_test_loss / (len(test_dataloader)))
    Test_Accuracy_list.append(total_test_acc / test_len)
    #
    if(total_test_acc / test_len) > min_acc:
        min_acc = total_test_acc / test_len
        res_TP_TN_FP_FN = TP_TN_FP_FN
        # torch.save(myModel.state_dict(), 'E:/博士成果/跟吴老师的第一篇文章/model/MI.pth')


    print("Epoch: {}/{} ".format(i + 1, epochs),
          "Training Loss: {:.4f} ".format(total_train_loss / len(train_dataloader)),
          "Training Accuracy: {:.4f} ".format(total_train_acc / train_len),
          "Test Loss: {:.4f} ".format(total_test_loss / len(test_dataloader)),
          "Test Accuracy: {:.4f}".format(total_test_acc / test_len)
          )


print(min_acc)
#混淆矩阵
f,ax2 = plt.subplots(figsize = (10, 8))

sns.heatmap(res_TP_TN_FP_FN,fmt='d',cmap='Blues',annot=True)

# ax2.set_title('confusion_matrix')
ax2.set_xlabel('Pred')
ax2.set_ylabel('True')

plt.show()
#保存
f.savefig('E:/博士成果/跟吴老师的第一篇文章/model/confusion_matrix.png', bbox_inches='tight')

# print("TP: {}".format(res_TP_TN_FP_FN[0]))
# print("TN: {}".format(res_TP_TN_FP_FN[1]))
# print("FP: {}".format(res_TP_TN_FP_FN[2]))
# print("FN: {}".format(res_TP_TN_FP_FN[3]))

train_x1 = range(0, 20000)
train_x2 = range(0, 20000)
train_y1 = Train_Accuracy_list
train_y2 = Train_Loss_list
plt.subplot(2, 1, 1)
plt.plot(train_x1, train_y1, 'o-')
plt.title('Train accuracy vs. epoches')
plt.ylabel('Train accuracy')
plt.subplot(2, 1, 2)
plt.plot(train_x2, train_y2, '.-')
plt.xlabel('Train loss vs. epoches')
plt.ylabel('Train loss')
plt.show()
#
test_x1 = range(0, 20000)
test_x2 = range(0, 20000)
test_y1 = Test_Accuracy_list
test_y2 = Test_Loss_list
plt.subplot(2, 1, 1)
plt.plot(test_x1, test_y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(test_x2, test_y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()