from matplotlib import pyplot as plt
import numpy as np
import sys
import cv2
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=sys.maxsize)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import datetime
from sklearn import mixture

tf.random.set_seed(44)

def _rescale(images):
    imgs = []
    for image in images:
        image = image.reshape( int(np.sqrt(images.shape[1])), int(np.sqrt(images.shape[1]))  )
        image = cv2.resize(image, (324, 324), interpolation=cv2.INTER_AREA)
        image = image.flatten()
        imgs.append(image)
    images = np.array(imgs)
    return images

data = np.genfromtxt('datasets/input_data.csv', delimiter=',')
labels = np.genfromtxt('datasets/input_labels.csv', delimiter=',')


X_train, X_test, y_train, y_test= train_test_split(data,labels, #random_state=44,
                                                   test_size=0.33)
X_train = _rescale(X_train)
X_test = _rescale(X_test)
X_train = X_train.reshape(-1,324,324,1)
X_test = X_test.reshape(-1,324,324,1)

autoencoder_model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(324,324,1)),
        tf.keras.layers.Conv2D(4, 3, activation=tf.nn.leaky_relu, strides=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(4, 3, activation=tf.nn.leaky_relu, strides=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(4, 4, activation=tf.nn.leaky_relu, strides=4, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Reshape([9, 9, 4]),
        tf.keras.layers.Conv2DTranspose(4, 4, strides=4, padding='same', activation=tf.nn.leaky_relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(4, 3, strides=3, padding='same', activation=tf.nn.leaky_relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(4, 3, strides=3, padding='same', activation=tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(1, (3, 3), padding='same'),
    ]
)

optimizer =Adam()


autoencoder_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),#'binary_crossentropy',
                          # loss='mse',
                          optimizer=optimizer,
                          metrics=["mae","mse"])


early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=10,
                                       verbose=1,
                                       mode='min')
checkpoint_path = "models/"+str(datetime.datetime.now().strftime("%H.%M.%S_%d.%m.%Y")+"/")
check_point = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 monitor='val_loss',
                                                 mode='min',
                                                 save_best_only=True)
epochs = 100
history = autoencoder_model.fit(X_train, X_train,
                                     batch_size=1000,
                                     epochs=epochs,
                                     validation_data=(X_test, X_test),
                                     callbacks=[early_stopping, check_point],
                                )

# plt.plot(history.history['loss'], label="train loss")
# plt.plot(history.history['val_loss'], label="validation loss")
# plt.title("Test Loss")
# plt.xlabel("Number of Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid()
# plt.show()


X_ = X_train
autoencoder_model.load_weights(checkpoint_path)


encoder = tf.keras.Model(autoencoder_model.input, autoencoder_model.layers[-8].output)
encoded_train = encoder.predict(X_train)
encoded_test = encoder.predict(X_test)
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(encoded_train)
lab_train = gmm.predict(encoded_train)
lab_test = gmm.predict(encoded_test)

tp,fp,fn,tn = 0,0,0,0
for p,t in zip(lab_train,y_train):
    if t and p:
        tp+=1
    elif t and not p:
        fn+=1
    elif not t and not p:
        tn+=1
    elif not t and p:
        fp+=1
print("tp,fp,fn,tn")
print(tp,fp,fn,tn)
print("train accuracy", (tp+tn)/(fp+fn+tp+tn)*100)

tp,fp,fn,tn = 0,0,0,0
for p,t in zip(lab_test,y_test):
    if t and p:
        tp+=1
    elif t and not p:
        fn+=1
    elif not t and not p:
        tn+=1
    elif not t and p:
        fp+=1
print("tp,fp,fn,tn")
print(tp,fp,fn,tn)
print("test accuracy", (tp+tn)/(fp+fn+tp+tn)*100)

y_test_xor = [int(not i) for i in y_test]
tp,fp,fn,tn = 0,0,0,0
for p,t in zip(lab_test,y_test_xor):
    if t and p:
        tp+=1
    elif t and not p:
        fn+=1
    elif not t and not p:
        tn+=1
    elif not t and p:
        fp+=1
print("alt")
print("tp,fp,fn,tn")
print(tp,fp,fn,tn)
print("test accuracy", (tp+tn)/(fp+fn+tp+tn)*100)

# Fun
preds = autoencoder_model.predict(X_, verbose=0)

for pred, true in zip(preds, X_):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 2, 1)
    plt.title("Reconstruction")
    pred = pred.reshape((324, 324))
    plt.imshow(pred, cmap="bone", label="train loss")
    ax = fig.add_subplot(1, 2, 2)
    true = true.reshape((324, 324))
    plt.imshow(true, cmap="bone", label="train loss")
    plt.title("Original")
    plt.show()
