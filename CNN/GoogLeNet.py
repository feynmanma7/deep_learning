#encoding:utf-8

from keras.model import Model
from keras.layers import Dense, Flatten, Dropout, 
	BatchNormalization, concatenet, Input
from keras.convolution import Conv2D, MaxPooling2D, 
	AveragePooling2D

import numpy as np 
np.random.seed(20170430)


def Conv2D_BN(x, 
	n_filter, 
	kernel_size, 
	padding='same', 
	strides=(1, 1)):

	x = Conv2D(n_filter, 
		kernel_size, 
		padding=padding, 
		strides=strides, 
		activation='relu')

	x = BatchNormalization(axis=3)(x)

	return x


def Inception(x, n_filter):
	branch1x1 = Conv2D_BN(x, n_filter, 
		kernel_size=(1, 1), 
		padding='same', 
		strides=(1, 1))

	branch3x3 = Conv2D_BN(x, n_filter, 
		kernel_size=(1, 1), 
		padding='same', 
		strides=(1, 1))
	branch3x3 = Conv2D_BN(branch3x3, n_filter, 
		kernel_size=(3, 3), 
		padding='same', 
		strides=(1, 1))

	branch5x5 = Conv2D_BN(x, n_filter, 
		kernel_size=(1, 1), 
		padding='same', 
		strides=(1, 1))
	branch5x5 = Conv2D_BN(branch5x5, n_filter, 
		kernel_size=(5, 5), 
		padding='same', 
		strides=(1, 1))

	branch_pool = MaxPooling2D(pool_size=(3, 3), 
		strides=(1, 1), 
		padding='same')(x)
	branch_pool = Conv2D_BN(branch_pool, n_filter, 
		kernel_size=(1, 1), 
		padding='same', 
		strides=(1, 1))

	x = concatenet([branch1x1, branch3x3, branch5x5, branch_pool])

	return x


# GoogLeNet for ImageNet, input: 224 * 224 * 3
def get_GoogLeNet_model():

	input_ = Input(shape=(224, 224, 3))

	# Conv-1, output: 64 * 112 * 112, 
	x = Conv2D_BN(64, (7, 7), 
			strides=(2, 2), 
			padding='same')(input_)

	# Pool-1, output: 64 * 56 * 56
	x = MaxPooling2D(pool_size=(3, 3), 
		strides=(2, 2), 
		padding='same')(x)

	# Conv-2, output: 192 * 56 * 56
	x = Conv2D_BN(192, (3, 3), 
		strides=(1, 1), 
		padding='same')(x)

	# Pool-2, output: 192 * 28 * 28
	x = MaxPooling2D(pool_size=(3, 3), 
		strides=(2, 2))(x)

	# Inception-1
	# output: (64 * 4) * 28 * 28 = 256 * 28 * 28
	x = Inception(x, 64)

	# output: 480 * 28 * 28
	x = Inception(x, 120)

	# output: 480 * 14 * 14
	x = MaxPooling2D(pool_size=(3, 3), 
		strides=(2, 2), 
		padding='same')(x)

	# Inception-2, output: _ * 14 * 14
	x = Inception(x, 128)(x)  # 512 * 14 * 14
	x = Inception(x, 128)(x)  # 512 * 14 * 14
	x = Inception(x, 128)(x)  # 512 * 14 * 14
	x = Inception(x, 132)(x)  # 528 * 14 * 14
	x = Inception(x, 208)(x)  # 832 * 14 * 14

	# output: 832 * 7 * 7
	x = MaxPooling2D(pool_size=(3, 3),
		strides=(2, 2), 
		padding='same')(x)

	# Inception-3, output: _ * 7 * 7
	x = Inception(x, 208)(x)  # 832 * 7 * 7
	x = Inception(x, 256)(x)  # 1024 * 7 * 7

	# output: 1024 * 1 * 1
	x = AveragePooling2D(pool_size=(7, 7), 
		strides=(7, 7), 
		padding='same')(x)

	x = Droput(0.4)(x)

	x = Dense(1000, activation='relu')(x)

	x = Dense(1000, activation='softmax')(x)

	model = Model(input_, x)

	model.compile(loss='categorical_crossentropy', 
		optimizer='sgd', 
		metrics=['accuracy'])

	return model



