#encoding:utf-8

"""
Reference: https://blog.keras.io/building-autoencoders-in-keras.html
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

encoding_dim = 32

input_img = Input(shape = (784, ))

# encoded = Dense(encoding_dim, activation = "relu")(input_img)
encoded = Dense(encoding_dim, activation = "relu",
    activity_regularizer = regularizers.l1(1e-4))(input_img)

decoded = Dense(784, activation = "sigmoid")(encoded)

# autoencoder
autoencoder = Model(inputs = input_img, outputs = decoded)

# train
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

from keras.datasets import mnist
import numpy as np

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


# encoder
encoder = Model(inputs = input_img, outputs = encoded)

# decoder
encoded_input = Input(shape = (encoding_dim, ))

decoder_layer = autoencoder.layers[-1]

decoder = Model(inputs = encoded_input, 
    outputs = decoder_layer(encoded_input))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt

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

plt.show()