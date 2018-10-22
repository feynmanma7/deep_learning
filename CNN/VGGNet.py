#encoding:utf-8

from keras.model import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.convolution import Conv2D, MaxPooling2D

import numpy as np 
np.random.seed(20170430)

def get_VGGNet13_model():
	model = Sequential()

	# Conv-1_1, output: 64 * 224 * 224
	model.add(Conv2D(64, (3, 3), 
		strides=(1, 1), 
		input_shape=(224, 224, 3), 
		padding='same', 
		activation='relu', 
		kernel_initializer='uniform'))

	# Conv-1_2, output: 64 * 224 * 224
	model.add(Conv2D(64, (3, 3), 
		strides=(1, 1), 
		padding='same', 
		activation='relu', 
		kernel_initializer='uniform'))

	# Pool-1, output: 64 * 112 * 112
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Conv-2_1, output: 128 * 112 * 112
	model.add(Conv2D(128, (3, 3), 
		strides=(1, 1), 
		padding='same', 
		activation='relu', 
		kernel_initializer='uniform'))

	# Conv-2_2, output: 128 * 112 * 112
	model.add(Conv2D(128, (3, 3), 
		strides=(1, 1), 
		padding='same', 
		activation='relu', 
		kernel_initializer='uniform'))

	# Pool-2, output: 128 * 56 * 56
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Conv-3_1, output: 256 * 56 * 56
	model.add(Conv2D(256, (3, 3), 
		strides=(1, 1), 
		padding='same', 
		activation='relu', 
		kernel_initializer='uniform'))

	# Conv-3_2, output: 256 * 56 * 56
	model.add(Conv2D(256, (3, 3), 
		strides=(1, 1), 
		padding='same', 
		activation='relu', 
		kernel_initializer='uniform'
		))

	# Pool-3, output: 256 * 28 * 28
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Conv-4_1, output: 512 * 28 * 28
	model.add(Conv2D(512, (3, 3), 
		strides=(1, 1), 
		padding='same', 
		activation='relu', 
		kernel_initializer='uniform'))

	# Conv-4_2, output: 512 * 28 * 28
	model.add(Conv2D(512, (3, 3), 
		strides=(1, 1), 
		padding='same', 
		activation='relu', 
		kernel_initializer='uniform'))

	# Pool-4, output: 512 * 14 * 14
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Conv-5_1, output: 512 * 28 * 28
	model.add(Conv2D(512, (3, 3), 
		strides=(1, 1), 
		padding='same', 
		activation='relu', 
		kernel_initializer='uniform'))

	# Conv-5_2, output: 512 * 28 * 28
	model.add(Conv2D(512, (3, 3), 
		strides=(1, 1), 
		padding='same', 
		activation='relu', 
		kernel_initializer='uniform'))

	# Pool-5, output: 512 * 14 * 14
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())

	model.add(Dense(4096, activation='relu'))

	model.add(Droput(0.5))

	model.add(Dense(4096, activation='relu'))

	model.add(Droput(0.5))

	model.add(Dense(1000, activation='softmax'))

	model.compile(loss='categorical_crossentropy', 
		optimizer='sgd', 
		metrics=['accuracy'])

	return model