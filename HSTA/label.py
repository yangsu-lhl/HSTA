import numpy as np
from scipy.io import loadmat

#取出每个被试的标签
"""
label = np.empty([0,1])
for j in range(6):

    data1 = loadmat("E:/博士成果/跟吴老师的第一篇文章/数据集/BCI_2a/BCI_C_IV_2a/A09T.mat")["data"][0][j+3]["y"][0, 0][:,0]
    # print(data1)

    label1 = np.empty([0,1])
    for i in range(48):
        label1 = np.vstack([label1, (np.repeat(data1[i], 3).reshape(-1,1))])

    print(label1.shape)
    label = np.vstack([label, label1])

print(label.shape)
np.save("E:/博士成果/跟吴老师的第一篇文章/label/S9T.npy", label)
"""

#合并被试标签

label = np.empty([0,1])
for i in range(9):
    data = np.load("E:/博士成果/跟吴老师的第一篇文章/label/S{}T.npy".format(i+1))
    label = np.vstack([label, data])

print(label.shape)
np.save("E:/博士成果/跟吴老师的第一篇文章/label/all_label.npy", label)