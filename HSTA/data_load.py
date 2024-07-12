import os
import numpy as np
from scipy.io import loadmat
import mne
# run = np.zeros((36000,22))
#取所有被试各run的数据
"""
for j in range(6):
    data1 = loadmat("E:/博士成果/跟吴老师的第一篇文章/数据集/BCI_2a/BCI_C_IV_2a/A04T.mat")["data"][0][j+1]["X"][0,0][:,0:22]  #(96735,22)
    # print(data1.shape)

    trial = loadmat("E:/博士成果/跟吴老师的第一篇文章/数据集/BCI_2a/BCI_C_IV_2a/A04T.mat")["data"][0][j+1]["trial"][0,0] #(48,1)
    # print(trial.shape)
    data = np.zeros((750,22))
    for i in range(48):
        trial_data = data1[trial[i,0] + 750 : trial[i,0] + 1500, :]
        data = np.concatenate([data, trial_data])

    effect_data = data[750:36750, :]
    print(effect_data.shape)
    np.save("E:/博士成果/跟吴老师的第一篇文章/data/raw/S4_T_run{}.npy".format(j+1), effect_data)
"""
#合并run
"""
for i in range(9):
    data1 = np.load("E:/博士成果/跟吴老师的第一篇文章/data/raw/S{}_T_run1.npy".format(i+1))
    data2 = np.load("E:/博士成果/跟吴老师的第一篇文章/data/raw/S{}_T_run2.npy".format(i+1))
    data3 = np.load("E:/博士成果/跟吴老师的第一篇文章/data/raw/S{}_T_run3.npy".format(i+1))
    data4 = np.load("E:/博士成果/跟吴老师的第一篇文章/data/raw/S{}_T_run4.npy".format(i+1))
    data5 = np.load("E:/博士成果/跟吴老师的第一篇文章/data/raw/S{}_T_run5.npy".format(i+1))
    data6 = np.load("E:/博士成果/跟吴老师的第一篇文章/data/raw/S{}_T_run6.npy".format(i+1))
    data = np.concatenate([data1, data2, data3, data4, data5, data6])
    print(data.shape)
    np.save("E:/博士成果/跟吴老师的第一篇文章/data/raw/S{}_T.npy".format(i+1), data)
"""
"""
# 读取Physionet数据集
import mne

# 读取edf文件
file_path = "D:\postgraduate\MIdata\physionet\S001\S001R06.edf"
raw = mne.io.read_raw_edf(file_path, preload=True)

# 打印数据信息
print("raw信息：")
print(raw.info)

# 获取原始数据
data, times = raw[:, :]
print("data:")
# print(data.shape) (64, 20000)
# print(data)
print("times:")
# print(times.shape) (20000,)
# print(times)
"""

"""
import mne
# 第4个任务,T1:2:左拳,T2:3:右拳
raw_data = mne.io.read_raw_edf('D:\postgraduate\MIdata\physionet\S001\S001R04.edf')
raw_data.plot()
raw_data.load_data() #载入数据
events_from_annot, event_dict = mne.events_from_annotations(raw_data)
print(event_dict)
print(events_from_annot)
# print(events_from_annot.shape) # (30,3)
# print(raw_data.info) # 数据信息（如通道信息、采样率等）
data, times = raw_data[:, :]
print("data:")
# print(data.shape) # （64， 20000）
print(data)
print(data[0][2000])
print(data[0][2001])
print(data[0][2639])
print(data[0][2640])
"""

# 共有109个被试
for x in range(109):

    # 因为第100个数据集不够，是坏数据，所以不要第100个数据
    if x == 99:
        continue

    # 只需要读取4,6,8,10,12,14这6个想象任务
    # 左拳:0000 右拳:1111 双拳:2222 双脚:3333 (打标签)

    # 第4个任务,T1:2:左拳,T2:3:右拳,--------------------------------------------------------

    # 读取EDF文件
    edf_file = "D:\postgraduate\MIdata\physionet\S{:03}\S{:03}R04.edf".format(x+1, x+1)
    raw = mne.io.read_raw_edf(edf_file, preload=True)

    # 获取事件和事件字典信息
    events, event_dict = mne.events_from_annotations(raw)

    # 定义感兴趣的两个事件类型的事件ID
    event_id_1 = event_dict['T1']  # 第一个事件类型的事件ID
    event_id_2 = event_dict['T2']  # 第二个事件类型的事件ID

    # 定义时间范围
    tmin = 0  # 事件开始的时间（秒）
    tmax = 4 - 1/160  # 事件结束的时间（秒）tmax=4的话,会多得到一个点,641个点

    # 选择 EEG 通道
    picks = mne.pick_types(raw.info, eeg=True)

    # 根据事件ID提取数据段
    epochs_event_1 = mne.Epochs(raw, events, event_id=event_id_1, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True) # 左拳 8个
    epochs_event_2 = mne.Epochs(raw, events, event_id=event_id_2, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True) # 右拳 7个

    # 获取两个事件类型的数据的Numpy数组
    data_array_event_1 = epochs_event_1.get_data(copy=False)  # (8, 64, 640)
    data_array_event_2 = epochs_event_2.get_data(copy=False)  # (7, 64, 640)
    # print("data_array_event_1.shape:")
    # print(data_array_event_1.shape)
    # print("data_array_event_2.shape:")
    # print(data_array_event_2.shape)

    # 获取两个事件的长度
    length_data_array_event_1 = data_array_event_1.shape[0]  # 8
    length_data_array_event_2 = data_array_event_2.shape[0]  # 7

    # 生成标签
    task_4_label = np.array([0, 0, 0, 0])
    for i in range(length_data_array_event_1 - 1):
        task_4_label = np.concatenate([task_4_label, [0, 0, 0, 0]])
    for i in range(length_data_array_event_2):
        task_4_label = np.concatenate([task_4_label, [1, 1, 1, 1]])

    # 合并两个事件类型的数据
    data_combined = np.concatenate([data_array_event_1, data_array_event_2], axis=0)
    # print(data_combined.shape)  # (15, 64, 640) 要变成(64, 640 * 15)=(64, 9600)
    # 将三维数组的子数组沿着第二维度（axis=1）进行拼接
    task_4_feature = np.concatenate(data_combined, axis=1)
    # print("task_4_feature:")
    # print(task_4_feature.shape)  # (64, 9600)
    # print(task_4_feature)

    # 保存为Numpy数组文件（.npy）这是第4个任务的脑电想象数据和标签
    # np.save("task_4_feature.npy", task_4_feature)
    # np.save("task_4_label.npy", task_4_label)
    # print("task_4_feature:")
    # print(task_4_feature.shape)  # (64, 9600)
    # print(task_4_feature)
    # print("task_4_label:")
    # print(task_4_label.shape)  # (60,)
    # print(task_4_label)

    # print("任务四数据提取成功")

    # 第6个任务,T1:2:双拳,T2:3:双脚,--------------------------------------------------------
    import mne
    import numpy as np

    # 读取EDF文件
    edf_file = "D:\postgraduate\MIdata\physionet\S{:03}\S{:03}R06.edf".format(x+1, x+1)
    raw = mne.io.read_raw_edf(edf_file, preload=True)

    # 获取事件和事件字典信息
    events, event_dict = mne.events_from_annotations(raw)

    # 定义感兴趣的两个事件类型的事件ID
    event_id_1 = event_dict['T1']  # 第一个事件类型的事件ID
    event_id_2 = event_dict['T2']  # 第二个事件类型的事件ID

    # 定义时间范围
    tmin = 0  # 事件开始的时间（秒）
    tmax = 4 - 1/160  # 事件结束的时间（秒）tmax=4的话,会多得到一个点,641个点

    # 选择 EEG 通道
    picks = mne.pick_types(raw.info, eeg=True)

    # 根据事件ID提取数据段
    epochs_event_1 = mne.Epochs(raw, events, event_id=event_id_1, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True) # 双拳 8个
    epochs_event_2 = mne.Epochs(raw, events, event_id=event_id_2, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True) # 双脚 7个

    # 获取两个事件类型的数据的Numpy数组
    data_array_event_1 = epochs_event_1.get_data(copy=False)  # (8, 64, 640)
    data_array_event_2 = epochs_event_2.get_data(copy=False)  # (7, 64, 640)
    # print("data_array_event_1.shape:")
    # print(data_array_event_1.shape)
    # print("data_array_event_2.shape:")
    # print(data_array_event_2.shape)

    # 获取两个事件的长度
    length_data_array_event_1 = data_array_event_1.shape[0]  # 8
    length_data_array_event_2 = data_array_event_2.shape[0]  # 7

    # 生成标签
    task_6_label = np.array([2, 2, 2, 2])
    for i in range(length_data_array_event_1 - 1):
        task_6_label = np.concatenate([task_6_label, [2, 2, 2, 2]])
    for i in range(length_data_array_event_2):
        task_6_label = np.concatenate([task_6_label, [3, 3, 3, 3]])

    # 合并两个事件类型的数据
    data_combined = np.concatenate([data_array_event_1, data_array_event_2], axis=0)
    # print(data_combined.shape)  # (15, 64, 640) 要变成(64, 640 * 15)=(64, 9600)
    # 将三维数组的子数组沿着第二维度（axis=1）进行拼接
    task_6_feature = np.concatenate(data_combined, axis=1)
    # print("task_4_feature:")
    # print(task_4_feature.shape)  # (64, 9600)
    # print(task_4_feature)

    # 保存为Numpy数组文件（.npy）这是第4个任务的脑电想象数据和标签
    # np.save("task_4_feature.npy", task_4_feature)
    # np.save("task_4_label.npy", task_4_label)
    # print("task_6_feature:")
    # print(task_6_feature.shape)  # (64, 9600)
    # print(task_6_feature)
    # print("task_6_label:")
    # print(task_6_label.shape)  # (60,)
    # print(task_6_label)

    # print("任务六数据提取成功")

    # 第8个任务,T1:2:左拳,T2:3:右拳,--------------------------------------------------------
    import mne
    import numpy as np

    # 读取EDF文件
    edf_file = "D:\postgraduate\MIdata\physionet\S{:03}\S{:03}R08.edf".format(x+1, x+1)
    raw = mne.io.read_raw_edf(edf_file, preload=True)

    # 获取事件和事件字典信息
    events, event_dict = mne.events_from_annotations(raw)

    # 定义感兴趣的两个事件类型的事件ID
    event_id_1 = event_dict['T1']  # 第一个事件类型的事件ID
    event_id_2 = event_dict['T2']  # 第二个事件类型的事件ID

    # 定义时间范围
    tmin = 0  # 事件开始的时间（秒）
    tmax = 4 - 1/160  # 事件结束的时间（秒）tmax=4的话,会多得到一个点,641个点

    # 选择 EEG 通道
    picks = mne.pick_types(raw.info, eeg=True)

    # 根据事件ID提取数据段
    epochs_event_1 = mne.Epochs(raw, events, event_id=event_id_1, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True) # 左拳 8个
    epochs_event_2 = mne.Epochs(raw, events, event_id=event_id_2, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True) # 右拳 7个

    # 获取两个事件类型的数据的Numpy数组
    data_array_event_1 = epochs_event_1.get_data(copy=False)  # (8, 64, 640)
    data_array_event_2 = epochs_event_2.get_data(copy=False)  # (7, 64, 640)
    # print("data_array_event_1.shape:")
    # print(data_array_event_1.shape)
    # print("data_array_event_2.shape:")
    # print(data_array_event_2.shape)

    # 获取两个事件的长度
    length_data_array_event_1 = data_array_event_1.shape[0]  # 8
    length_data_array_event_2 = data_array_event_2.shape[0]  # 7

    # 生成标签
    task_8_label = np.array([0, 0, 0, 0])
    for i in range(length_data_array_event_1 - 1):
        task_8_label = np.concatenate([task_8_label, [0, 0, 0, 0]])
    for i in range(length_data_array_event_2):
        task_8_label = np.concatenate([task_8_label, [1, 1, 1, 1]])

    # 合并两个事件类型的数据
    data_combined = np.concatenate([data_array_event_1, data_array_event_2], axis=0)
    # print(data_combined.shape)  # (15, 64, 640) 要变成(64, 640 * 15)=(64, 9600)
    # 将三维数组的子数组沿着第二维度（axis=1）进行拼接
    task_8_feature = np.concatenate(data_combined, axis=1)
    # print("task_4_feature:")
    # print(task_4_feature.shape)  # (64, 9600)
    # print(task_4_feature)

    # 保存为Numpy数组文件（.npy）这是第4个任务的脑电想象数据和标签
    # np.save("task_4_feature.npy", task_4_feature)
    # np.save("task_4_label.npy", task_4_label)
    # print("task_8_feature:")
    # print(task_8_feature.shape)  # (64, 9600)
    # print(task_8_feature)
    # print("task_8_label:")
    # print(task_8_label.shape)  # (60,)
    # print(task_8_label)

    # print("任务八数据提取成功")

    # 第10个任务,T1:2:双拳,T2:3:双脚,--------------------------------------------------------
    import mne
    import numpy as np

    # 读取EDF文件
    edf_file = "D:\postgraduate\MIdata\physionet\S{:03}\S{:03}R10.edf".format(x+1, x+1)
    raw = mne.io.read_raw_edf(edf_file, preload=True)

    # 获取事件和事件字典信息
    events, event_dict = mne.events_from_annotations(raw)

    # 定义感兴趣的两个事件类型的事件ID
    event_id_1 = event_dict['T1']  # 第一个事件类型的事件ID
    event_id_2 = event_dict['T2']  # 第二个事件类型的事件ID

    # 定义时间范围
    tmin = 0  # 事件开始的时间（秒）
    tmax = 4 - 1/160  # 事件结束的时间（秒）tmax=4的话,会多得到一个点,641个点

    # 选择 EEG 通道
    picks = mne.pick_types(raw.info, eeg=True)

    # 根据事件ID提取数据段
    epochs_event_1 = mne.Epochs(raw, events, event_id=event_id_1, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True) # 双拳 8个
    epochs_event_2 = mne.Epochs(raw, events, event_id=event_id_2, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True) # 双脚 7个

    # 获取两个事件类型的数据的Numpy数组
    data_array_event_1 = epochs_event_1.get_data(copy=False)  # (8, 64, 640)
    data_array_event_2 = epochs_event_2.get_data(copy=False)  # (7, 64, 640)
    # print("data_array_event_1.shape:")
    # print(data_array_event_1.shape)
    # print("data_array_event_2.shape:")
    # print(data_array_event_2.shape)

    # 获取两个事件的长度
    length_data_array_event_1 = data_array_event_1.shape[0]  # 8
    length_data_array_event_2 = data_array_event_2.shape[0]  # 7

    # 生成标签
    task_10_label = np.array([2, 2, 2, 2])
    for i in range(length_data_array_event_1 - 1):
        task_10_label = np.concatenate([task_10_label, [2, 2, 2, 2]])
    for i in range(length_data_array_event_2):
        task_10_label = np.concatenate([task_10_label, [3, 3, 3, 3]])

    # 合并两个事件类型的数据
    data_combined = np.concatenate([data_array_event_1, data_array_event_2], axis=0)
    # print(data_combined.shape)  # (15, 64, 640) 要变成(64, 640 * 15)=(64, 9600)
    # 将三维数组的子数组沿着第二维度（axis=1）进行拼接
    task_10_feature = np.concatenate(data_combined, axis=1)
    # print("task_4_feature:")
    # print(task_4_feature.shape)  # (64, 9600)
    # print(task_4_feature)

    # 保存为Numpy数组文件（.npy）这是第4个任务的脑电想象数据和标签
    # np.save("task_4_feature.npy", task_4_feature)
    # np.save("task_4_label.npy", task_4_label)
    # print("task_10_feature:")
    # print(task_10_feature.shape)  # (64, 9600)
    # print(task_10_feature)
    # print("task_10_label:")
    # print(task_10_label.shape)  # (60,)
    # print(task_10_label)

    # print("任务十数据提取成功")

    # 第12个任务,T1:2:左拳,T2:3:右拳,--------------------------------------------------------
    import mne
    import numpy as np

    # 读取EDF文件
    edf_file = "D:\postgraduate\MIdata\physionet\S{:03}\S{:03}R12.edf".format(x+1, x+1)
    raw = mne.io.read_raw_edf(edf_file, preload=True)

    # 获取事件和事件字典信息
    events, event_dict = mne.events_from_annotations(raw)

    # 定义感兴趣的两个事件类型的事件ID
    event_id_1 = event_dict['T1']  # 第一个事件类型的事件ID
    event_id_2 = event_dict['T2']  # 第二个事件类型的事件ID

    # 定义时间范围
    tmin = 0  # 事件开始的时间（秒）
    tmax = 4 - 1/160  # 事件结束的时间（秒）tmax=4的话,会多得到一个点,641个点

    # 选择 EEG 通道
    picks = mne.pick_types(raw.info, eeg=True)

    # 根据事件ID提取数据段
    epochs_event_1 = mne.Epochs(raw, events, event_id=event_id_1, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True) # 左拳 8个
    epochs_event_2 = mne.Epochs(raw, events, event_id=event_id_2, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True) # 右拳 7个

    # 获取两个事件类型的数据的Numpy数组
    data_array_event_1 = epochs_event_1.get_data(copy=False)  # (8, 64, 640)
    data_array_event_2 = epochs_event_2.get_data(copy=False)  # (7, 64, 640)
    # print("data_array_event_1.shape:")
    # print(data_array_event_1.shape)
    # print("data_array_event_2.shape:")
    # print(data_array_event_2.shape)

    # 获取两个事件的长度
    length_data_array_event_1 = data_array_event_1.shape[0]  # 8
    length_data_array_event_2 = data_array_event_2.shape[0]  # 7

    # 生成标签
    task_12_label = np.array([0, 0, 0, 0])
    for i in range(length_data_array_event_1 - 1):
        task_12_label = np.concatenate([task_12_label, [0, 0, 0, 0]])
    for i in range(length_data_array_event_2):
        task_12_label = np.concatenate([task_12_label, [1, 1, 1, 1]])

    # 合并两个事件类型的数据
    data_combined = np.concatenate([data_array_event_1, data_array_event_2], axis=0)
    # print(data_combined.shape)  # (15, 64, 640) 要变成(64, 640 * 15)=(64, 9600)
    # 将三维数组的子数组沿着第二维度（axis=1）进行拼接
    task_12_feature = np.concatenate(data_combined, axis=1)
    # print("task_4_feature:")
    # print(task_4_feature.shape)  # (64, 9600)
    # print(task_4_feature)

    # 保存为Numpy数组文件（.npy）这是第4个任务的脑电想象数据和标签
    # np.save("task_4_feature.npy", task_4_feature)
    # np.save("task_4_label.npy", task_4_label)
    # print("task_12_feature:")
    # print(task_12_feature.shape)  # (64, 9600)
    # print(task_12_feature)
    # print("task_12_label:")
    # print(task_12_label.shape)  # (60,)
    # print(task_12_label)

    # print("任务十二数据提取成功")

    # 第14个任务,T1:2:双拳,T2:3:双脚,--------------------------------------------------------
    import mne
    import numpy as np

    # 读取EDF文件
    edf_file = "D:\postgraduate\MIdata\physionet\S{:03}\S{:03}R14.edf".format(x+1, x+1)
    raw = mne.io.read_raw_edf(edf_file, preload=True)

    # 获取事件和事件字典信息
    events, event_dict = mne.events_from_annotations(raw)

    # 定义感兴趣的两个事件类型的事件ID
    event_id_1 = event_dict['T1']  # 第一个事件类型的事件ID
    event_id_2 = event_dict['T2']  # 第二个事件类型的事件ID

    # 定义时间范围
    tmin = 0  # 事件开始的时间（秒）
    tmax = 4 - 1/160  # 事件结束的时间（秒）tmax=4的话,会多得到一个点,641个点

    # 选择 EEG 通道
    picks = mne.pick_types(raw.info, eeg=True)

    # 根据事件ID提取数据段
    epochs_event_1 = mne.Epochs(raw, events, event_id=event_id_1, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True) # 双拳 8个
    epochs_event_2 = mne.Epochs(raw, events, event_id=event_id_2, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True) # 双脚 7个

    # 获取两个事件类型的数据的Numpy数组
    data_array_event_1 = epochs_event_1.get_data(copy=False)  # (8, 64, 640)
    data_array_event_2 = epochs_event_2.get_data(copy=False)  # (7, 64, 640)
    # print("data_array_event_1.shape:")
    # print(data_array_event_1.shape)
    # print("data_array_event_2.shape:")
    # print(data_array_event_2.shape)

    # 获取两个事件的长度
    length_data_array_event_1 = data_array_event_1.shape[0]  # 8
    length_data_array_event_2 = data_array_event_2.shape[0]  # 7

    # 生成标签
    task_14_label = np.array([2, 2, 2, 2])
    for i in range(length_data_array_event_1 - 1):
        task_14_label = np.concatenate([task_14_label, [2, 2, 2, 2]])
    for i in range(length_data_array_event_2):
        task_14_label = np.concatenate([task_14_label, [3, 3, 3, 3]])

    # 合并两个事件类型的数据
    data_combined = np.concatenate([data_array_event_1, data_array_event_2], axis=0)
    # print(data_combined.shape)  # (15, 64, 640) 要变成(64, 640 * 15)=(64, 9600)
    # 将三维数组的子数组沿着第二维度（axis=1）进行拼接
    task_14_feature = np.concatenate(data_combined, axis=1)
    # print("task_4_feature:")
    # print(task_4_feature.shape)  # (64, 9600)
    # print(task_4_feature)

    # 保存为Numpy数组文件（.npy）这是第4个任务的脑电想象数据和标签
    # np.save("task_4_feature.npy", task_4_feature)
    # np.save("task_4_label.npy", task_4_label)
    # print("task_14_feature:")
    # print(task_14_feature.shape)  # (64, 9600)
    # print(task_14_feature)
    # print("task_14_label:")
    # print(task_14_label.shape)  # (60,)
    # print(task_14_label)

    # print("任务十四数据提取成功")

    # 把所有任务的特征和标签合并一起
    feature = np.concatenate((task_4_feature, task_6_feature, task_8_feature, task_10_feature, task_12_feature, task_14_feature), axis=1)
    label = np.concatenate((task_4_label, task_6_label, task_8_label, task_10_label, task_12_label, task_14_label))
    print(feature.shape)
    # print(feature)
    print(label.shape)
    # print(label)
    # 保存为Numpy数组文件（.npy）这是所有任务的脑电想象数据和标签
    feature_save_folder = "D:\postgraduate\MIdata\physionet\data_intercept\Feature"
    label_save_folder = "D:\postgraduate\MIdata\physionet\data_intercept\Label"
    feature_save_path = os.path.join(feature_save_folder, "feature{:03}.npy".format(x + 1))
    label_save_path = os.path.join(label_save_folder, "label{:03}.npy".format(x + 1))
    # np.save("feature{:03}.npy".format(x+1), feature)
    # np.save("label{:03}.npy".format(x+1), label)
    # np.save(feature_save_path, feature)
    # np.save(label_save_path, label)
    print("第{}个被试数据提取成功---------------------------------------------------------".format(x + 1))
