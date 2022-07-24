
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import metrics
import pandas as pd
import random
import cv2
import time

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

def cut_blank(data_, tresh=0):
    gr = np.gradient(data_)
    i = 0
    for n in gr:
        if n > tresh:
            return i
        i += 1

def rescale(data_,len):
    divisor = int(np.floor(data_.shape[0]/len))
    rescaled  = data_[1::divisor]
    return rescaled


failed = []
def on_pick(event):
    event.artist.remove()
    fig.canvas.draw()
    ind = event.artist
    id = re.search(r"_line\d+", str(ind))
    id = re.search(r"\d+", id.group())
    print("Plot no.: "+str(id.group())+" marked as FAILED")
    failed.append(int(id.group()))


start = time.time()

input_dir = "input_data/" 

listOfFiles = os.listdir(input_dir)

fig, ax = plt.subplots()
runs = dict()
i = 0
for file in listOfFiles:
    try:
        print(i, file)
        data = pd.read_csv(input_dir + file)
        var_name = data.columns[1]
        t_series = data[var_name].values
        del data
        ts_start = cut_blank(t_series,)
        t_series = t_series[ts_start:]
        if t_series.shape[0] < 300:
            print("TOO SHORT, len: " + str(t_series.shape[0]))
            continue
        runs[i] = t_series
        ax.plot(runs[i], picker=5)
        i += 1

    except:
        pass

fig.canvas.callbacks.connect('pick_event', on_pick)
plt.title(str(input_dir[:-1])+"\nClick on FAILED runs\nClose the window when done")
plt.show()

labels = [0 if a in failed else 1 for a in range(i)]
labels = np.array(labels)
try:
    os.mkdir("datasets/" + input_dir[:-1])
except:
    print("dir already exists, or something")
    pass

filename_l = "datasets/" + input_dir[:-1] + "/" + input_dir[:-1]+"_"+var_name+"_lab.csv"
print("Saving labels... " + str(labels.shape[0]) + " samples")
np.savetxt(filename_l, labels, delimiter=",")
print("Saved file as: "+ filename_l)

unique, counts = np.unique(labels, return_counts=True)
label_occurrence = dict(zip(unique, counts))

label_occurrence_rate = (label_occurrence.get(0)/labels.shape[0], label_occurrence.get(1)/labels.shape[0] )
print("Labels rate",label_occurrence_rate)
equalized_0_label_rate = 0.5
min_ts_len = 300
img_len = 324

# if label_occurrence_rate[0] < label_occurrence_rate[1]: 
#     equalizer_multipier = ( round(labels.shape[0]*equalized_0_label_rate/label_occurrence.get(0)) ,0)
# elif label_occurrence_rate[0] > label_occurrence_rate[1]:
#     equalizer_multipier = (0, round(labels.shape[0] * (1-equalized_0_label_rate) / label_occurrence.get(0)))

equalizer_multipier =(8,0)# (8,0)

equalized_labels = []
i = 0
for file,lab_ in zip(listOfFiles,labels):
    print(i, file,"label",lab_)
    data = pd.read_csv(input_dir + file)
    var_name = data.columns[1]
    t_series = data[var_name].values
    t = data[data.columns[0]].values
    del data

    #cut the begining
    ts_start = cut_blank(t_series,)
    t_series = t_series[ts_start:]
    t = t[ts_start:]

    #ignore if too short
    if t_series.shape[0] < min_ts_len:
        print("TOO SHORT, len: " + str(t_series.shape[0]))
        continue

    # #interpolate
    # t_range = [a for a in range(max(t) + 1)]
    # if not np.array_equal(t, t_range):
    #     print(t)
    #     print(t_range)
    #     t_series1 = np.interp(t_range, t, t_series)
    #     print("przed:",t_series.shape[0], "po ",t_series1.shape[0],(t_series1.shape[0] - t_series.shape[0]) )

    #rescale before more complicated operations
    t_series = rescale(t_series,img_len)

    #recurrence
    t_series = np.gradient(t_series)
    t_series = t_series.reshape(-1, 1)
    rp = recurrence_plot_signed(t_series)
    rp = cv2.resize(rp, (img_len, img_len), interpolation=cv2.INTER_AREA)
    arr = rp.ravel()

    equalized_labels.append(lab_)

    if i == 0:
        recursive_dataset = np.array(arr)
        if lab_ == 0:
            for eqs in range(equalizer_multipier[0]):
                t_aug = np.array([n + random.randrange(-1, 2) * np.random.rand() * n * 0.001 for n in t_series])
                t_aug = t_aug.reshape(-1, 1)
                rp = recurrence_plot_signed(t_aug)
                rp = cv2.resize(rp, (img_len, img_len), interpolation=cv2.INTER_AREA)
                arr = rp.ravel()
                equalized_labels.append(0)  # append additional label
                recursive_dataset = np.vstack((recursive_dataset, arr))
        elif lab_ == 1:
            for eqs in range(equalizer_multipier[1]):
                t_aug = np.array([n + random.randrange(-1, 2) * np.random.rand() * n * 0.001 for n in t_series])
                t_aug = t_aug.reshape(-1, 1)
                rp = recurrence_plot_signed(t_aug)
                rp = cv2.resize(rp, (img_len, img_len), interpolation=cv2.INTER_AREA)
                arr = rp.ravel()
                equalized_labels.append(1)  # append additional label
                recursive_dataset = np.vstack((recursive_dataset, arr))
        i += 1
        continue

    i += 1
    recursive_dataset = np.vstack((recursive_dataset, arr))

    #equalize + add noise
    # if lab_ == (0 if label_occurrence_rate[0] else 0):
    if lab_ == 0:
        for eqs in range(equalizer_multipier[0]):
            t_aug = np.array([n + random.randrange(-1, 2) * np.random.rand() * n * 0.001 for n in t_series])
            t_aug = t_aug.reshape(-1, 1)
            rp = recurrence_plot_signed(t_aug)
            rp = cv2.resize(rp, (img_len, img_len), interpolation=cv2.INTER_AREA)
            arr_add = rp.ravel()
            equalized_labels.append(0)#append additional label
            recursive_dataset = np.vstack((recursive_dataset, arr_add))
    elif lab_ == 1:
        for eqs in range(equalizer_multipier[1]):
            t_aug = np.array([n + random.randrange(-1, 2) * np.random.rand() * n * 0.001 for n in t_series])
            t_aug = t_aug.reshape(-1, 1)
            rp1 = recurrence_plot_signed(t_aug)
            rp1 = cv2.resize(rp, (img_len, img_len), interpolation=cv2.INTER_AREA)
            arr_add = rp.ravel()
            equalized_labels.append(1)#append additional label
            recursive_dataset = np.vstack((recursive_dataset, arr_add))



if equalizer_multipier[0] or equalizer_multipier[1]:

    unique, counts = np.unique(equalized_labels, return_counts=True)
    label_occurrence = dict(zip(unique, counts))
    label_occurrence_rate = (label_occurrence.get(0) / labels.shape[0], label_occurrence.get(1) / labels.shape[0])

    equalized_labels = np.array(equalized_labels)
    filename_l_eq = "datasets/" + input_dir[:-1] + "/"+  input_dir[:-1]+"_"+var_name+"_eq0-"+str(equalized_labels[0])+"_lab.csv"
    print("Saving equalized labels... "+ str(equalized_labels.shape[0]) + " samples")
    np.savetxt(filename_l_eq, equalized_labels, delimiter=",")
    print("Saved file as: "+ filename_l)

    filename =  "datasets/" + input_dir[:-1] + "/"+ input_dir[:-1]+"_"+var_name+"_eq0-"+str(equalized_labels[0])+"_rp.csv"
    print("Saving equalized rp... "+ str(recursive_dataset.shape[0]) + " samples")
    np.savetxt(filename, recursive_dataset, delimiter=",")
    print("Saved file as: "+filename)
else:
    filename = "datasets/" + input_dir[:-1] + "/" + input_dir[:-1]+"_"+var_name+"_rp.csv"
    print("Saving rp... "+ str(recursive_dataset.shape[0]) + " samples")
    np.savetxt(filename, recursive_dataset, delimiter=",")
    print("Saved file as: "+filename)

end = time.time()
print("Program finished in t: " + str(end - start) +" s")
