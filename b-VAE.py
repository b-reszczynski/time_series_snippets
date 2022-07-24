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
import tensorflow as tf
import datetime
import sklearn as sk
tf.random.set_seed(44)
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions
from tensorflow.keras import Sequential

data = np.genfromtxt('input_img.csv', delimiter=',')


X_train, X_test = train_test_split(data, random_state=44,  test_size=0.33)
X_train = X_train.reshape(-1,315,315,1)
X_test = X_test.reshape(-1,315,315,1)

# encoded_size = 152#16
base_depth = 32
input_shape = (315,315,1)

# input_shape = datasets_info.features['image'].shape
encoded_size = 32
base_depth = 32

prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)
tfpl = tfp.layers
encoder = Sequential([
    tfkl.InputLayer(input_shape=input_shape),
    tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
    tfkl.Conv2D(base_depth, 3, strides=3,
                padding='same', activation=tf.nn.leaky_relu),

    tfkl.Conv2D(base_depth, 3, strides=3,
                padding='same', activation=tf.nn.leaky_relu),

    tfkl.Conv2D(1, 3, strides=1,
                padding='same', activation=tf.nn.leaky_relu),
    tfkl.Flatten(),
    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
               activation=None),
    tfpl.MultivariateNormalTriL(
        encoded_size,
        activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=N),
])
# Encoder
encoder.summary()
decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[560]),
    tfkl.Reshape([35, 35, 32]),#tfkl.Reshape([1, 1, encoded_size]),
    tfkl.Conv2DTranspose(base_depth, 3, strides=3,
                         padding='same', activation=tf.nn.leaky_relu),

    tfkl.Conv2DTranspose(base_depth, 3, strides=3,
                         padding='same', activation=tf.nn.leaky_relu),

    tfkl.Conv2D(filters=1, kernel_size=3, strides=1,
                padding='same', activation=None),
    tfkl.Flatten(),
    tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
])
# Decoder
decoder.summary()

vae = tfk.Model(inputs=encoder.inputs,
                outputs=decoder(encoder.outputs[0]))

negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
            loss=negative_log_likelihood)


print("\nvae\n")
vae.summary()

early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
checkpoint_path = "models/"+str(datetime.datetime.now().strftime("%H.%M.%S_%d.%m.%Y")+"/")
check_point = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 monitor='val_loss',
                                                 mode='min',
                                                 save_best_only=True)
history = vae.fit(X_train, X_train,
                                     # batch_size=62,
                                     epochs=1,
                                     validation_data=(X_test, X_test),
                                     callbacks=[early_stopping,check_point],
                                )

# vae.load_weights(checkpoint_path)
plt.plot(history.history['loss'], label="train loss")
plt.plot(history.history['val_loss'], label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
X_ = X_train
preds = vae.predict(X_, verbose=0)

for pred, true in zip(preds, X_):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 2, 1)
    plt.title("Reconstruction")
    pred = pred.reshape((315, 315))
    plt.imshow(pred, cmap="bone", label="train loss")
    ax = fig.add_subplot(1, 2, 2)
    true = true.reshape((315, 315))
    plt.imshow(true, cmap="bone", label="train loss")
    plt.title("Oryginal")
    plt.show()




