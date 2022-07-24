import bz2
import os
import re
from shutil import copyfile
from bz2 import BZ2File
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import metrics
import pandas as pd
import cv2


def recurrence_plot(s, eps=None, steps=None):
    if eps == None: eps = 10  # 0.1
    if steps == None: steps = 1000  # 10
    d = sk.metrics.pairwise.pairwise_distances(s)
    d = np.floor(d / eps)
    d[d > steps] = steps
    return d / steps


def recurrence_plot_signed(s):
    rp = metrics.pairwise_distances(s)
    rp_max = np.max(rp)
    rp_min = np.min(rp)

    for i in range(1, rp.shape[0]):
        for j in range(1, rp.shape[0]):
            rp[i][j] = rp[i][j] * np.sign(s[i - 1] - s[i])
            rp[i][j] = (rp[i][j] - rp_min) / rp_max
    rp = np.delete(rp, 0, axis=0)
    rp = np.delete(rp, 0, axis=1)
    return rp

def start_of_test(data_, tresh=0):
    gr = np.gradient(data_)
    i = 0
    for n in gr:
        if n > tresh:
            return i
        i += 1

def rescale(data_):
    divisor = int(np.floor(len(data_)/324))
    rescaled  = data_[1::divisor]
    print(divisor,len(rescaled) )
    return rescaled

img_len = 324
input_dir = "input_dir"/"
# labels = np.genfromtxt('input_labels.csv', delimiter=',')
listOfFiles = os.listdir(input_dir)

i = 0

for file in listOfFiles:
    try:
        print(i, file)
        data = pd.read_csv(input_dir + file)
        var_name = data.columns[1]
        t_series = data[var_name].values
        del data
        ts_start = start_of_test(t_series,)
        t_series = t_series[ts_start:]
        if t_series.shape[0] < 300:
            print("TOO SHORT")
            continue
        t_series = rescale(t_series)
        t_series = np.gradient(t_series)
        t_series = t_series.reshape(-1, 1)
        # t_series = np.resize(t_series,324)
        # rp = recurrence_plot(t_series, 10, 1000)
        # rp = recurrence_plot(t_series, 10000, 10000)#ST_CMXB_RESTART_EVO_BAT
        rp = recurrence_plot_signed(t_series)
        rp = cv2.resize(rp, (img_len, img_len), interpolation=cv2.INTER_AREA)

        # investigate
        # fig = plt.figure(figsize=(8, 8))
        # fig.add_subplot(1, 2, 1)
        # plt.imshow(rp, cmap="bone")
        # fig.add_subplot(1, 2, 2)
        # plt.plot(t_series)
        # plt.title(labels[i])
        plt.imshow(rp1, cmap="bone")
        # plt.show()

        arr = rp.ravel()
        if i == 0:
            recursive_dataset = np.array(arr)
            # recursive_dataset = np.vstack((recursive_dataset, arr))
            # recursive_dataset = np.vstack((recursive_dataset, arr))
            i += 1
            continue

        i += 1
        recursive_dataset = np.vstack((recursive_dataset, arr))
        # recursive_dataset = np.vstack((recursive_dataset, arr))
        # recursive_dataset = np.vstack((recursive_dataset, arr))
        # if i >= 2:
        #     break
    except:
        pass
filename = input_dir[:-1]+"_"+var_name+"_rp.csv"
print("Saving file...")
print(recursive_dataset.shape)
np.savetxt(filename, recursive_dataset, delimiter=",")
print("Saved file as: "+filename)