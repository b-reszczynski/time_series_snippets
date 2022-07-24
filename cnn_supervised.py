"""IMPORTS"""
from matplotlib import pyplot as plt
import datetime
import time
import numpy as np
import sys
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
import kwakwa_lib as kwa
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
tf.random.set_seed(44)
import sklearn, matplotlib, pandas

"""LOAD DATASET"""
img_len = 324

start = time.time()
data = np.genfromtxt('datasets/input_data.csv', delimiter=',')
labels = np.genfromtxt('datasets/input_labels.csv', delimiter=',')
print(data.shape)
print(labels.shape)

"""DATASET SPLIT & RESHAPE"""
X_train, X_test, y_train, y_test= train_test_split(data,labels,
                                                   random_state=45,
                                                   test_size=0.30)
img_dim = [-1,img_len,img_len,1]
X_train = X_train.reshape(img_dim)
X_test = X_test.reshape(img_dim)

assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]
"""PLOT LABELED DATA"""
# for img,lab in zip(data,labels ):
#     fig = plt.figure(figsize=(15, 14))
#     img = img.reshape((324, 324))
#     plt.imshow(img, cmap="bone")
#     plt.title(lab)
#     plt.show()

"""MODEL"""
lambda_ = 0.2
model = Sequential([
Input(shape=(img_len,img_len,1)),
Conv2D(64, (3, 3),  activation = 'relu'),
MaxPooling2D(pool_size = (2, 2)),
Conv2D(32, (3, 3), activation = 'relu'),
MaxPooling2D(pool_size = (2, 2)),
Conv2D(16, (3, 3), activation = 'relu'),
MaxPooling2D(pool_size = (2, 2)),
Flatten(),

Dense(units = 8, activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
Dense(units = 1, activation = 'sigmoid',kernel_regularizer=tf.keras.regularizers.l2(lambda_))
])

checkpoint_path = "models/"+str(datetime.datetime.now().strftime("%H.%M.%S_%d.%m.%Y")+"/")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  # min_delta=0,
                                                  patience=10,
                                                  verbose=1,
                                                  mode='min')

check_point = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 monitor='val_accuracy',
                                                 mode='max',
                                                 save_best_only=True)


###################################################CROSS VALIDATION

# def make_dataset(X_data,y_data,n_splits):
#
#     def gen():
#         for train_index, test_index in KFold(n_splits).split(X_data):
#             X_train, X_test = X_data[train_index], X_data[test_index]
#             y_train, y_test = y_data[train_index], y_data[test_index]
#             yield X_train,y_train,X_test,y_test
#
#     return tf.data.Dataset.from_generator(gen, (tf.float64,tf.float64,tf.float64,tf.float64))
#
# data, labels = shuffle(data, labels)
# dataset=make_dataset(data,labels,10)
# f1 = []
# acc = []
# for X_train,y_train,X_test,y_test in dataset:
#
#
#     model.compile(loss='binary_crossentropy',
#                   optimizer="adam",
#                   metrics=["accuracy"])
#
#     X_train = tf.reshape(X_train, img_dim)
#     X_test = tf.reshape(X_test, img_dim)
#     # print(y_test)
#     model.fit(X_train, y_train,
#                                # batch_size=62,
#                                epochs=50,#17,16 val_accuracy: 0.9516
#                                validation_data=(X_test, y_test),
#                                callbacks=[early_stopping, check_point],
#                                verbose=1
#                           )
#
#     model.load_weights(checkpoint_path)
#     y_pred = model.predict(X_test, verbose=0)
#     f1_score = kwa.f1_score(y_test, y_pred.round())
#     acc_score = kwa.accuracy_score(y_test,y_pred.round())
#     f1.append(f1_score)
#     acc.append(acc_score)
#     print("BEST  n:",len(f1), "f1:", round(f1_score, 3),"accuracy: ", round(acc_score, 3))
#     tn, fp, fn, tp = kwa.confusion_matrix(y_test, y_pred.round()).ravel()
#     print("BEST TNR:", round(tn * 100 / (tn + fn), 3), ", TPR Precission:", round(tp * 100 / (tp + fp), 3), ", Recall:",
#           round(tp * 100 / (tp + fn), 3))
# print("============================")
# print(f1)
# print(acc)
# print("f1 mean: ", sum(f1)/len(f1), "f1 variance: ",statistics.stdev(f1))
# print("acc mean: ", sum(acc)/len(acc), "f1 variance: ",statistics.stdev(acc))
# print("f1",f1)
# print("acc",acc)
# print("============================")

###################################################

optimizer = tf.keras.optimizers.Adam()
model.compile(loss='binary_crossentropy',
              optimizer="adam",
              # optimizer=optimizer,
              metrics=["accuracy"])

history = model.fit(X_train, y_train,
                                     batch_size=1000,
                                     epochs=100,
                                     validation_data=(X_test, y_test),
                                     callbacks=[early_stopping, check_point],
                                )

"""PLOT TRAIN HISTORY"""
fig = plt.figure(figsize=(8, 8))
fig.add_subplot(1, 2, 1)
plt.plot(history.history['loss'], label="train loss")
plt.plot(history.history['val_loss'], label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.1)
fig.add_subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label="train accuracy")
plt.plot(history.history['val_accuracy'], label="validation accuracy")
plt.title("Test accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.1)
plt.show()

"""PLOT FILTERS"""
# for mod in model.layers:
#     try:
#         filters, biases = mod.get_weights()
#         f_min, f_max = filters.min(), filters.max()
#         filters = (filters - f_min) / (f_max - f_min)
#         n_filters, ix = 6, 1
#         for i in range(n_filters):
#             f = filters[:, :, :, i]
#             for j in range(3):
#                 ax = plt.subplot(n_filters, 3, ix)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 plt.imshow(f[:, :, j], cmap='gray')
#                 ix += 1
#         plt.show()
#     except:
#         # print("except: ",mod)
#         pass

"""PREDICT"""
model.load_weights(checkpoint_path)
y_pred = model.predict(X_test, verbose=0)
# print(y_test)


"""HISTOGRAM"""
y1 = []
y0 = []
for i in range(y_pred.shape[0]):
    if y_pred[i] > 0.5:
        y1.append(y_pred[i])
    else:
        y0.append(y_pred[i])
plt.hist(y1, bins=5)
plt.hist(y0, bins=5)
plt.legend(["1","0"])
plt.grid(alpha=0.1)
plt.show()
print("Ans 0 max p", max(y0))
print("Ans 1 min p",min(y1))
"""SHOW METRICS"""
for i in range(y_pred.shape[0]):
    if y_pred[i] > 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
tn, fp, fn, tp = kwa.confusion_matrix(y_test, y_pred).ravel()
print("TEST TN:", round(tn * 100 / (tn + fn), 3), ", TP:", round(tp * 100 / (tp + fp), 3), ", f1:", round(kwa.f1_score(y_test, y_pred), 3))
print("TEST Tp:", tp, ", Tn:", tn)
y_pred_train = model.predict(X_train, verbose=0)
for i in range(y_pred_train.shape[0]):
    if y_pred_train[i] > 0.5:
        y_pred_train[i] = 1
    else:
        y_pred_train[i] = 0
tn, fp, fn, tp = kwa.confusion_matrix(y_train, y_pred_train).ravel()
print("TRAIN TN:", round(tn * 100 / (tn + fn), 3), ", TP:", round(tp * 100 / (tp + fp), 3), ", f1:", round(kwa.f1_score(y_train, y_pred_train), 3))
print("TRAIN Tp:", tp, ", Tn:", tn)
"""SAVE MODEL"""
# tf.saved_model.save(model,str("models/"+"model_tp"+str(int(tp * 100 / (tp + fp)))+"tn"+str(int(tn * 100 / (tn + fn)))+"_"+str(datetime.datetime.now().strftime("%H.%M.%S_%d.%m.%Y"))  ))

end = time.time()
print("Program finished in t: " + str(end - start) +" s")
