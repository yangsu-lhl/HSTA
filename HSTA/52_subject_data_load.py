import scipy.io
import numpy as np
import os

# 共有52个被试
for x in range(1):

    # 替换为你的.mat文件路径 这只是一个被试aa的数据
    mat_file_path = 'D:/postgraduate/MIdata/52_subject/s{:02}.mat'.format(x+1)

    # 使用scipy.io.loadmat()加载.mat文件
    mat_data = scipy.io.loadmat(mat_file_path)

    # 查看.mat文件中的所有变量
    # print("Variables in the .mat file:", mat_data.keys())

    # 获取特定变量的数据，例如变量名为 'your_variable'
    eeg = mat_data['eeg']
    # print("eeg:", eeg)

    # 处理左手 左手标签：000

    # 先取出左手数据
    imagery_left = eeg[0][0][7]  # (68, 358400) 7 x 512 x 100 或 # (68, 430080) 7 x 512 x 120
    # 怎么取出EEG数据 变成64通道 [:64, :] 因为数据文档没用说明 4个EMG数据应该是最后4个通道 现在并不确定
    imagery_left = imagery_left[:64, :]
    length = imagery_left.shape[1]
    # 取出每7个数据点的中间3个数据点
    # 将形状为 [68, 358400]或[68, 430080] 的数组分成几个形状为 [68, 3584] 的子数组
    split_imagery_left = np.split(imagery_left, length // 3584, axis=1)  # (68, 3584)
    # print(split_imagery_left.)
    # 取出中间3，4，5秒的数据
    sub_split_imagery_left = [sub_array[:, 1024:2560] for sub_array in split_imagery_left]

    # 将子数组拼接起来
    concat_sub_split_imagery_left = np.concatenate(sub_split_imagery_left, axis=1)
    # print(concat_sub_split_imagery_left[0])
    # print(concat_sub_split_imagery_left[0].shape)
    # 打印数组的形状和内容
    print(concat_sub_split_imagery_left.shape)  # (64, 153600) 153600/512 = 300

    # # 得到左手标签 生成100个0的一维数组
    # if length == 358400:
    #     label_left = np.zeros(100)
    # else:
    #     label_left = np.zeros(120)
    #
    # # 打印结果数组
    # # print(label_left)
    # print(label_left.shape)  # (300,)


    # # 处理右手 右手标签：111
    #
    # # 先取出右手数据
    # imagery_right = eeg[0][0][8]  # (68, 358400) 7 x 512 x 10
    # # 变成64通道 [:64, :]
    # imagery_right = imagery_right[:64, :]
    # length = imagery_right.shape[1]
    # # 取出每7个数据点的中间3个数据点
    # # 将形状为 [68, 358400] 的数组分成几个形状为 [68, 7] 的子数组
    # split_imagery_right = np.split(imagery_right, length // 3584, axis=1)  # (68, 3584)
    # # print(split_imagery_left.)
    # # 取出中间3，4，5秒的数据
    # sub_split_imagery_right = [sub_array[:, 1024:2560] for sub_array in split_imagery_right]
    #
    # # 将子数组拼接起来得到形状为 [68, 358400] 的数组
    # concat_sub_split_imagery_right = np.concatenate(sub_split_imagery_right, axis=1)
    #
    # # 打印数组的形状和内容
    # # print(concat_sub_split_imagery_right.shape)  # (68, 153600) 153600/512 = 300
    # # print(concat_sub_split_imagery_right)
    #
    # # 得到右手标签 生成3 x 100个1的一维数组
    # if length == 358400:
    #     label_right = np.ones(100)
    # else:
    #     label_right = np.ones(120)
    #
    # # 打印结果数组
    # # print(label_right)
    # print(label_right.shape)  # (100,)
    #
    # # 合并两只手
    #
    # # 合并特征
    # # feature = np.concatenate((concat_sub_split_imagery_left, concat_sub_split_imagery_right), axis=1)
    # feature = concat_sub_split_imagery_right
    # # print(feature.shape)
    #
    # # 合并标签
    # # label = np.concatenate([label_left, label_right])
    # label = label_right
    # # print(label.shape)
    #
    # # 保存为Numpy数组文件（.npy）这是所有任务的脑电想象数据和标签
    # feature_save_folder = "D:/postgraduate/MIdata/52_subject/right_feature"
    # label_save_folder = "D:/postgraduate/MIdata/52_subject/right_label"
    #
    # feature_save_path = os.path.join(feature_save_folder, "right_feature{:02}.npy".format(x + 1))
    # label_save_path = os.path.join(label_save_folder, "right_label{:02}.npy".format(x + 1))
    #
    # # np.save(feature_save_path, feature)
    # # np.save(label_save_path, label)
    # print("第{}个被试数据提取成功,并保存为.npy文件".format(x + 1))
    # # print(feature.shape, label)
