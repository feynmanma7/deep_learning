#encoding:utf-8

"""
Reference: https://blog.keras.io/building-autoencoders-in-keras.html
"""
from model_util import create_vae_model
import numpy as np
np.random.seed(20170430)

original_dim = 784
latent_dim = 2

vae, vae_generator = create_vae_model()

from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data(path='mnist.npz')

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape( (len(x_train), np.prod(x_train.shape[1:])) )
x_test = x_test.reshape( (len(x_test), np.prod(x_test.shape[1:])) )

vae.fit(x_train, x_train,
    shuffle=True, 
    epochs=100,
    batch_size=128,
    validation_data=(x_test, x_test))


vae.save_weights('../model/vae_weights.h5')
