#encoding:utf-8

from keras.models import Model
from keras.layers import add, Flatten, Dense, Dropout, BatchNormalization
from keras.convolution import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D

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


def Conv_Block(input_, 
	n_filter, 
	kernel_size, 
	strides=(1, 1), 
	with_conv_shortcut):

	x = Conv2D_BN(input_, n_filter, 
		kernel_size=kernel_size, 
		strides=strides, 
		padding='same')

	x = Conv2D_BN(x, n_filter, 
		kernel_size=kernel_size, 
		padding='same')

	if with_conv_shortcut:
		shortcut = Conv2D_BN(input_, 
			n_filter, 
			kernel_size=kernel_size, 
			strides=strides)

		x = add([x, shortcut])
		return x

	else:
		x = add([x, input])
		return x


# Resnet-34
def get_ResNet_model():

	input_ = Input(224, 224, 3)

	# output: 227 * 227 * 3
	x = ZeroPadding2D((3, 3))(input_)

	# output: 64 * 111 * 111 , ceil: (227 - 7 + 1) / 2
	x = Conv2D_BN(x, n_filter=64, 
		kernel_size=(7, 7),
		strides=(2, 2), 
		padding='valid')

	# output: 64 * 56 * 56
	x = MaxPooling2D(pool_size=(3, 3), 
		strides=(2, 2), 
		padding='same')(x)

	# Conv_Block, output: 64 * 56 * 56
	x = Conv_Block(x, n_filter=64, kernel_size=(3, 3))
	x = Conv_Block(x, n_filter=64, kernel_size=(3, 3))
	x = Conv_Block(x, n_filter=64, kernel_size=(3, 3))

	# output: 128 * 28 * 28
	x = Conv_Block(x, n_filter=128, kernel_size=(3, 3), 
		strides=(2, 2), with_conv_shortcut=True)

	# output: 128 * 28 * 28
	x = Conv_Block(x, n_filter=128, kernel_size=(3, 3))
	x = Conv_Block(x, n_filter=128, kernel_size=(3, 3))
	x = Conv_Block(x, n_filter=128, kernel_size=(3, 3))

	# output: 256 * 14 * 14
	x = Conv_Block(x, n_filter=256, kernel_size=(3, 3), 
		strides=(2, 2), with_conv_shortcut=True)

	# output: 256 * 14 * 14
	x = Conv_Block(x, n_filter=256, kernel_size=(3, 3))
	x = Conv_Block(x, n_filter=256, kernel_size=(3, 3))
	x = Conv_Block(x, n_filter=256, kernel_size=(3, 3))
	x = Conv_Block(x, n_filter=256, kernel_size=(3, 3))

	# output: 512 * 7 * 7
	x = Conv_Block(x, n_filter=512, kernel_size=(3, 3), 
		strides=(2, 2), with_conv_shortcut=True)

	# output: 512 * 7 * 7
	x = Conv_Block(x, n_filter=512, kernel_size=(3, 3))
	x = Conv_Block(x, n_filter=512, kernel_size=(3, 3))

	# output: 512 * 1 * 1
	x = AveragePooling2D(pool_size=(7, 7))(x)

	x = Flatten()(x)

	x = Dense(1000, activation='softmax')(x)

	model = Model(inputs=input_, outputs=x)

	model.compile(loss='categorical_crossentropy', 
		optimizer='sgd', 
		metrics=['accuracy'])

	return model


