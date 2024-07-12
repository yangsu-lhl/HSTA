import math
import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter

'''
file name: DE_3D_Feature
input: the path of saw EEG file in SEED-VIG dataset
output: the 3D feature of all subjects
'''

# step1: input raw data
# step2: decompose frequency bands
# step3: calculate DE
# step4: stack them into 3D featrue


def butter_bandpass(low_freq, high_freq, fs, order=5):
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# calculate DE
def calculate_DE(saw_EEG_signal):
    variance = np.var(saw_EEG_signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


# filter for 5 frequency bands
def butter_bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_bandpass(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


def decompose_to_DE(file):
    # read data  sample * channel [1416000, 17]
    data = np.load(file)
    data = data.transpose([1, 0])
    # sampling rate
    frequency = 512
    # samples 1416000
    samples = data.shape[0]
    # 100 samples = 1 DE
    num_sample = int(samples/512)
    channels = data.shape[1]
    bands = 5
    # init DE [141600, 17, 5]
    DE_3D_feature = np.empty([num_sample, channels, bands])

    temp_de = np.empty([0, num_sample])

    for channel in range(channels):
        trial_signal = data[:, channel]
        # get 5 frequency bands
        delta = butter_bandpass_filter(trial_signal, 1, 4,   frequency, order=3)
        theta = butter_bandpass_filter(trial_signal, 4, 8,   frequency, order=3)
        alpha = butter_bandpass_filter(trial_signal, 8, 12,  frequency, order=3)
        beta  = butter_bandpass_filter(trial_signal, 12, 30, frequency, order=3)
        gamma = butter_bandpass_filter(trial_signal, 30, 50, frequency, order=3)
        # DE
        DE_delta = np.zeros(shape=[0], dtype=float)
        DE_theta = np.zeros(shape=[0], dtype=float)
        DE_alpha = np.zeros(shape=[0], dtype=float)
        DE_beta = np.zeros(shape=[0], dtype=float)
        DE_gamma = np.zeros(shape=[0], dtype=float)
        # DE of delta, theta, alpha, beta and gamma
        for index in range(num_sample):
            DE_delta = np.append(DE_delta, calculate_DE(delta[index * 512:(index + 1) * 512]))
            DE_theta = np.append(DE_theta, calculate_DE(theta[index * 512:(index + 1) * 512]))
            DE_alpha = np.append(DE_alpha, calculate_DE(alpha[index * 512:(index + 1) * 512]))
            DE_beta  = np.append(DE_beta,  calculate_DE(beta[index * 512:(index + 1) * 512]))
            DE_gamma = np.append(DE_gamma, calculate_DE(gamma[index * 512:(index + 1) * 512]))
        temp_de = np.vstack([temp_de, DE_delta])
        temp_de = np.vstack([temp_de, DE_theta])
        temp_de = np.vstack([temp_de, DE_alpha])
        temp_de = np.vstack([temp_de, DE_beta])
        temp_de = np.vstack([temp_de, DE_gamma])

    temp_trial_de = temp_de.reshape(-1, 5, num_sample)
    temp_trial_de = temp_trial_de.transpose([2, 0, 1])
    DE_3D_feature = np.vstack([temp_trial_de])

    # print(DE_3D_feature.shape)  # (300, 64, 5)
    subarray_shape = (3, 64, 5)
    subarrays = DE_3D_feature.reshape(-1, *subarray_shape)  # (100, 3, 64, 5)
    subarrays_transpose = subarrays.transpose([0, 2, 3, 1])
    # print(subarrays_transpose.shape)  # (100, 64, 5, 3)
    return subarrays_transpose


if __name__ == '__main__':
    # Fill in your SEED-VIG dataset path
    filePath = 'D:\postgraduate\MIdata/52_subject/right_feature/'
    dataName = os.listdir(filePath)

    X = np.empty([0, 60, 5])

    for i in range(len(dataName)):
        dataFile = filePath + dataName[i]
        print('processing {}'.format(dataName[i]))
        # every subject DE feature
        DE_feature = decompose_to_DE(dataFile)

        np.save("D:\postgraduate\MIdata/52_subject/right_feature_DE/S{:02}.npy".format(i + 1), DE_feature)
        # all subjects
        # X = np.vstack([X, DE_feature])

    # save .npy file
    # np.save("E:/博士成果/跟吴老师的第一篇文章/BCI-III-IIIa/DE/All_Subject.npy", X)