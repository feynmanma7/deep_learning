#encoding:utf-8
from keras.models import load_model
from scipy.stats import norm
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from model_util import create_vae_model, create_vcae_model

def vae_test():
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


def vcae_test():
	vcae_model, vcae_generator = create_vcae_model()

	vcae_model.load_weights('../model/vcae_weights.h5')

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


if __name__ == '__main__':
	vcae_test()





