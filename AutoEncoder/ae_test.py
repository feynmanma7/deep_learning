#encoding:utf-8
from keras.models import load_model
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from model_util import create_vae_model

vae_model, vae_generator = create_vae_model()

vae_model.load_weights('../model/vae_weights.h5')

n = 15
figure = np.zeros((28 * 15, 28 * 15))

grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

epsilon_std = 1.0
for i, yi in enumerate(grid_x):
	for j, xi in enumerate(grid_y):
		z_sample = np.array([[xi, yi]]) * epsilon_std
		x_decoded = vae_generator.predict(z_sample)
		digit = x_decoded[0].reshape(28, 28)
		figure[i * 28: (i+1) * 28,
			   j * 28: (j+1) * 28] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.savefig('../pic/vae.png')
plt.close()







