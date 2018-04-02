#encoding:utf-8

"""
Reference: https://blog.keras.io/building-autoencoders-in-keras.html
"""
from keras.layers import Input, Dense, Flatten, Lambda, \
		Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Reshape
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras import metrics
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

#print(vcae.summary())
vcae.fit(x_train, 
    shuffle=True, 
    epochs=1,
    batch_size=100,
    validation_data=(x_test, None))

from scipy.stats import norm
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')

n = 15
figure = np.zeros((28 * 15, 28 * 15))

grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

epsilon_std = 1.0
for i, yi in enumerate(grid_x):
	for j, xi in enumerate(grid_y):
		z_sample = np.array([[xi, yi]]) * epsilon_std
		x_decoded = vcae_generator.predict(z_sample)
		digit = x_decoded[0].reshape(28, 28)
		figure[i * 28: (i+1) * 28,
			   j * 28: (j+1) * 28] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.savefig('../pic/vcae.png')
plt.close()

vcae.save_weights('../model/vcae_weights.h5')
