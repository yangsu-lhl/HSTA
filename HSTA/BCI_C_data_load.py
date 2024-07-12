import scipy.io
import numpy as np

# 替换为你的.mat文件路径 这只是一个被试aa的数据
mat_file_path = 'D:/postgraduate/MIdata/BCI_Competition_Ⅳa/data_set_IVa_aa_mat/1000Hz/data_set_IVa_aa.mat'

# 使用scipy.io.loadmat()加载.mat文件
mat_data = scipy.io.loadmat(mat_file_path)

# 查看.mat文件中的所有变量
# print("Variables in the .mat file:", mat_data.keys())

# 获取特定变量的数据，例如变量名为 'your_variable'
__header__ = mat_data['__header__']
__version__ = mat_data['__version__']
__globals__ = mat_data['__globals__']
cnt = mat_data['cnt']  # 连续的脑电图信号，大小为[时间 x 通道]。
info = mat_data['nfo']  # info：结构提供具有字段的附加信息
mrk = mat_data['mrk']  # mrk：目标提示信息的结构与字段
# print(cnt.shape)  # (2984598, 118) (时间 x 通道)
cnt_transpose = cnt.transpose()  # 转置后 (118, 2984598)
feature = mrk[0][0][0][0]  # 想象特征数组 (280,)
label = mrk[0][0][1][0]  # 标签数组 (280,)
# 现在要找mark。。。info里面
# print(info[0][0])
print(info[0][0][2].shape)  # (1, 118)
print(info[0][0])
# print(info[0][0][2][0][1])
# print(info[0][0][2][0][1].dtype)  # 这个不是mark，是通道标识符
