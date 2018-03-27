#encoding:utf-8

from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras import regularizers
from keras import backend as K
from keras import metrics

import numpy as np

def vae_loss(x, x_decoded_mean):
    # Loss
    xent_loss = 784 * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)

epsilon_std = 1.0

#model = load_model('../model/vae.h5')

original_dim = 784
x = Input(shape=(original_dim, ))
h = Dense(256, activation='relu')(x)
z_mean = Dense(2)(h)
z_log_var = Dense(2)(h)

decoder_h = Dense(256, activation='relu')
decoder_mean = Dense(784, activation='sigmoid')

epsilon_std = 1.0

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], 2), 
        mean=0, 
        stddev=1.0)

    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(2, ))([z_mean, z_log_var])

h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# Model
vae = Model(inputs = x, outputs = x_decoded_mean)

def vae_loss(x, x_decoded_mean):
    # Loss
    xent_loss = 784 * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)

vae.compile(optimizer = 'rmsprop', loss=vae_loss)

vae.load_weights('../model/vae_weights.h5')

from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data(path='mnist.npz')

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape( (len(x_train), np.prod(x_train.shape[1:])) )
x_test = x_test.reshape( (len(x_test), np.prod(x_test.shape[1:])) )

print(vae.predict(x_test))