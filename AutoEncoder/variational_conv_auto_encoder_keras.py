#encoding:utf-8

"""
Reference: https://blog.keras.io/building-autoencoders-in-keras.html
"""
from keras.datasets import mnist
from model_util import create_vcae_model
import numpy as np
np.random.seed(20170430)

original_dim = 784
latent_dim = 2

vcae, vcae_generator = create_vcae_model()


(x_train, _), (x_test, _) = mnist.load_data(path='mnist.npz')

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape( (len(x_train), 28, 28, 1) )
x_test = x_test.reshape( (len(x_test), 28, 28, 1) )

vcae.fit(x_train, x_train,
    shuffle=True, 
    epochs=1,
    batch_size=128,
    validation_data=(x_test, x_test))

vcae.save_weights('../model/vcae_weights.h5')
