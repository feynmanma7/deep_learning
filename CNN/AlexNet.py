#encoding:utf-8

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layer.convolution import Conv2D, MaxPooling2D
import numpy as np 
np.random.seed(20170430)

# AlexNet for ImageNet
def get_AlexNet_model():
	model = Sequential()

	# Conv-1, ouput: 96 * 55 * 55, (227 - 11) / 4 + 1 
	model.add(Con2D(96, (11, 11), 
		strides=(4, 4), 
		input_shape=(227, 227, 3), 
		padding='valid', 
		activation='relu', 
		kernel_initializer='uniform'
		))

	# Pool-1, output: 96 * 27 * 27, (55 - 3) / 2 + 1
	model.add(MaxPooling2D(pool_size=(3, 3), 
		strides=(2, 2)))

	# Conv-2, output: 256 * 27 * 27
	model.add(Conv2D(256, (5, 5), 
		strides=(1, 1), 
		padding='same', 
		activation='relu', 
		kernel_initializer='uniform'))

	# Pool-2, output: 256 * 13 * 13, (27 - 3) / 2 + 1
	model.add(MaxPooling2D(pool_size=(3, 3), 
		strides=(2, 2)))

	# Conv-3, output: 384 * 13 * 13
	model.add(Conv2D(384, (3, 3), 
		strides=(1, 1), 
		padding='same', 
		activation='relu', 
		kernel_initializer='uniform'
		))

	# Conv-4, output: 384 * 13 * 13
	model.add(Conv2D(384, (3, 3),
		strides=(1, 1), 
		padding='same',
		activation='relu', 
		kernel_initializer='uniform'
		))

	# Conv-5, output: 256 * 13 * 13
	model.add(Conv2D(256, (3, 3), 
		strides=(1, 1), 
		padding='same',
		activation='relu',
		kernel_initializer='uniform'
		))

	# Pool-3, output: 256 * 6 * 6, (13 - 3) / 2 + 1
	model.add(MaxPooling2D(pool_size=(3, 3), 
		strides=(2, 2)))

	model.add(Flatten())

	mode.add(Dense(4096, activation='relu'))

	model.add(Dropout(0.5))

	model.add(Dense(4096, activation='relu'))

	model.add(Dropout(0.5))

	model.add(Dense(1000, activation='softmax'))

	model.compile(loss='categorical_crossentropy', 
		optimizer='sgd', 
		metrics=['accuracy'])

	return model