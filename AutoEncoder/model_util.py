#encoding:utf-8

from keras.layers import Input, Dense, Flatten, Lambda, \
		Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Reshape
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras import metrics

def sampling(args):
	    z_mean, z_log_var = args
	    epsilon = K.random_normal(
	        shape=(K.shape(z_mean)[0], 2), 
	        mean=0, 
	        stddev=1.0)

	    return z_mean + K.exp(z_log_var) * epsilon

def create_vcae_model():
	# variational convolutional auto-encoder

	def vcae_loss(x_input, x_output):
	    xent_loss = 784 * metrics.binary_crossentropy(
	    	K.flatten(x_input), K.flatten(x_output))
	    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	    return K.mean(xent_loss + kl_loss)

	epsilon_std = 1.0

	original_dim = 784
	x = Input(shape=(28, 28, 1))

	filters = 64

	# Encode
	e1 = Conv2D(1, 
		kernel_size=(2, 2),
		padding='same', activation='relu')(x)
	e2 = Conv2D(filters, 
		kernel_size=(2, 2), 
		padding='same', activation='relu',
		strides=(2,2))(e1)
	e3 = Conv2D(filters,
		kernel_size=3, 
		padding='same', activation='relu',
		strides=1)(e2)
	e4 = Conv2D(filters,
		kernel_size=3,
		padding='same', activation='relu',
		strides=1)(e3)

	flat = Flatten()(e4)

	h = Dense(128, activation='relu')(flat)
	
	# Sampling	
	z_mean = Dense(2)(h)
	z_log_var = Dense(2)(h)

	z = Lambda(sampling, output_shape=(2, ))([z_mean, z_log_var])

	# Decode
	d1 = Dense(128, activation='relu')(z)
	d2 = Dense(filters * 14 * 14, activation='relu')(d1) # upsampling
	d3 = Reshape((14, 14, filters))(d2) 
	d4 = Conv2DTranspose(filters, 
		kernel_size=3,
		padding='same',
		strides=1,
		activation='relu')(d3)
	d5 = Conv2DTranspose(filters,
	 	kernel_size=3,
	 	padding='same',
	 	strides=1, 
	 	activation='relu')(d4)
	d6 = Conv2DTranspose(filters,
	 	kernel_size=(3, 3),
	 	padding='valid',
	 	strides=(2, 2),
	 	activation='relu')(d5) #upsampling

	x_output = Conv2D(1,
		kernel_size=2,
		padding='valid',
		activation='sigmoid')(d6)

	# Generator, can name each decoder layer and reuse the name here.
	decoder_input = Input(shape=(2, ))
	 
	g1 = Dense(128, activation='relu')(decoder_input)
	g2 = Dense(filters * 14 * 14, activation='relu')(g1) # upsampling
	g3 = Reshape((14, 14, filters))(g2) 
	g4 = Conv2DTranspose(filters, 
		kernel_size=3,
		padding='same',
		strides=1,
		activation='relu')(g3)
	g5 = Conv2DTranspose(filters,
	 	kernel_size=3,
	 	padding='same',
	 	strides=1, 
	 	activation='relu')(g4)
	g6 = Conv2DTranspose(filters,
	 	kernel_size=(3, 3),
	 	padding='valid',
	 	strides=(2, 2),
	 	activation='relu')(g5) #upsampling

	decoder_output = Conv2D(1,
		kernel_size=2,
		padding='valid',
		activation='sigmoid')(g6)

	vcae_generator = Model(decoder_input, decoder_output)

	# Model
	vcae = Model(inputs=x, outputs=x_output)
	vcae.compile(optimizer='rmsprop', loss=vcae_loss)

	return vcae, vcae_generator


def create_vae_model():
	# variational auto-encoder

	def vae_loss(x, x_decoded_mean):
	    # Loss
	    xent_loss = 784 * metrics.binary_crossentropy(x, x_decoded_mean)
	    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	    return K.mean(xent_loss + kl_loss)

	epsilon_std = 1.0

	original_dim = 784
	x = Input(shape=(original_dim, ))


	h = Dense(256, activation='relu')(x)
	z_mean = Dense(2)(h)
	z_log_var = Dense(2)(h)

	decoder_h = Dense(256, activation='relu')
	decoder_mean = Dense(784, activation='sigmoid')

	epsilon_std = 1.0
	

	z = Lambda(sampling, output_shape=(2, ))([z_mean, z_log_var])

	h_decoded = decoder_h(z)
	x_decoded_mean = decoder_mean(h_decoded)

	# Model
	vae = Model(inputs = x, outputs = x_decoded_mean)
	vae.compile(optimizer = 'rmsprop', loss=vae_loss)

	# generator
	decoder_input = Input(shape=(2, ))
	_h_decoded = decoder_h(decoder_input)
	_x_decoded_mean = decoder_mean(_h_decoded)
	generator = Model(decoder_input, _x_decoded_mean)

	return vae, generator