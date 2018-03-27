#encoding:utf-8

"""
Reference: https://blog.keras.io/building-autoencoders-in-keras.html
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import numpy as np
np.random.seed(20170430)

encoding_dim = 32

input_img = Input(shape = (784, ))

encoded = Dense(128, activation = "relu")(input_img)
encoded = Dense(64, activation = "relu")(encoded)
encoded = Dense(32, activation = "relu")(encoded)

decoded = Dense(64, activation = "relu")(encoded)
decoded = Dense(128, activation = "relu")(decoded)
decoded = Dense(784, activation = "sigmoid")(decoded)

# autoencoder
autoencoder = Model(inputs = input_img, outputs = decoded)

# train
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

from keras.datasets import mnist


(x_train, _), (x_test, _) = mnist.load_data(path='mnist.npz')

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train,
                epochs = 100,
                batch_size = 256,
                shuffle = True,
                validation_data = (x_test, x_test))


decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

plt.switch_backend('agg')

n = 10
plt.figure(figsize = (20, 4))

for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('../tmp/dae.png')
plt.close()