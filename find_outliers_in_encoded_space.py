import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.datasets import elec_equip as ds
import numpy as np
import numpy as np
from scipy.spatial.distance import pdist, squareform
import numpy as np
import sys
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=sys.maxsize)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
import tensorflow as tf
import sklearn as sk
tf.random.set_seed(44)

data = np.genfromtxt('datasets/input_data.csv', delimiter=',')
labels = np.genfromtxt('datasets/input_labels.csv', delimiter=',')



min_max_scaler = sk.preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)
# print(data)
X_train, X_test, y_train, y_test = train_test_split(data.T, labelki,random_state=44,  test_size=0.33)
print(X_train.shape)

autoencoder_model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(316,)),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.Dense(150, activation=tf.nn.relu),
        tf.keras.layers.Dense(60, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(60, activation=tf.nn.relu),
        tf.keras.layers.Dense(150, activation=tf.nn.relu),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.Dense(316, activation=tf.keras.activations.linear),
    ]
)

optimizer =Adam(lr=0.001)
autoencoder_model.compile(loss='mae', optimizer=optimizer, metrics=["mae","mse"])
# print(autoencoder_model.summary())



early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
history = autoencoder_model.fit(X_train, X_train,
                                     batch_size=62,
                                     epochs=100,
                                     validation_data=(X_test, X_test),
                                     callbacks=[early_stopping],
                                )

plt.plot(history.history['loss'], label="train loss")
plt.plot(history.history['val_loss'], label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()


X1 = []
X0 = []
for x,y in zip(X_test,y_test):
    if y == 1:
        X1.append(x)
    else:
        X0.append(x)
X0 = np.array(X0)
X1 = np.array(X1)
encoder = tf.keras.Model(autoencoder_model.input, autoencoder_model.layers[-5].output)
encoded_train = encoder.predict(X_train)
encoded_test = encoder.predict(X_test)
encoded_test1 = encoder.predict(X1)
encoded_test0 = encoder.predict(X0)

print(encoded_train)

clf = IsolationForest(random_state=0).fit(encoded_train)
clf_all = clf.predict(encoded_test)
print("all",clf_all)
clf_1 = clf.predict(encoded_test1)
print("TRUE",clf_1)
clf_a0 = clf.predict(encoded_test0)
print("FALSE",clf_a0)


preds = autoencoder_model.predict(X_test, verbose=0)



