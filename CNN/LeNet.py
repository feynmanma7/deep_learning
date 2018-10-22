#encoding:utf-8

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

import numpy as np 
np.random.seed(20170430)


# LeNet for MNIST
def get_LeNet_model():
	model = Sequential()

	# Conv-1, output: 32 * 24 * 24
	model.add(Conv2D(32, (5, 5), 
		strides=(1, 1), 
		input_shape=(28, 28, 1), 
		padding='valid', 
		activation='relu', 
		kernel_initializer='uniform'))

	# Pool-1, output: 32 * 12 * 12
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Conv-2, output: 64 * 8 * 8
	model.add( Conv2D(64, (5, 5), 
		strides=(1, 1), 
		padding='valid', 
		activation='relu', 
		kernel_initializer='uniform'))

	# Pool-2, output: 64 * 4 * 4
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())

	model.add(Dense(100, activation='relu'))

	model.add(Dense(10, activation='softmax'))

	modle.compile(optimizer='sgd',
		loss='categorical_crossentropy', 
		metrics=['accuracy'])

	return model

