#encoding:utf-8

from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras import regularizers
from keras import backend as K
from keras import metrics

import matplotlib.pyplot as plt
plt.switch_backend('agg')

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

# generator
decoder_input = Input(shape=(2, ))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

n = 15
figure = np.zeros((28 * 15, 28 * 15))

grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
	for j, xi in enumerate(grid_y):
		z_sample = np.array([[xi, yi]]) * epsilon_std
		x_decoded = generator.predict(z_sample)
		digit = x_decoded[0].reshape(28, 28)
		figure[i * 28: (i+1) * 28,
			   j * 28: (j+1) * 28] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.savefig('../pic/vae.png')
plt.close()







