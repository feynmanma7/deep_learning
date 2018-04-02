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
	
	# input image dimensions
	img_rows, img_cols, img_chns = 28, 28, 1
	# number of convolutional filters to use
	filters = 64
	# convolution kernel size
	num_conv = 3

	batch_size = 100
	if K.image_data_format() == 'channels_first':
	    original_img_size = (img_chns, img_rows, img_cols)
	else:
	    original_img_size = (img_rows, img_cols, img_chns)
	latent_dim = 2
	intermediate_dim = 128
	epsilon_std = 1.0
	epochs = 5

	x = Input(shape=original_img_size)
	conv_1 = Conv2D(img_chns,
	                kernel_size=(2, 2),
	                padding='same', activation='relu')(x)
	conv_2 = Conv2D(filters,
	                kernel_size=(2, 2),
	                padding='same', activation='relu',
	                strides=(2, 2))(conv_1)
	conv_3 = Conv2D(filters,
	                kernel_size=num_conv,
	                padding='same', activation='relu',
	                strides=1)(conv_2)
	conv_4 = Conv2D(filters,
	                kernel_size=num_conv,
	                padding='same', activation='relu',
	                strides=1)(conv_3)
	flat = Flatten()(conv_4)
	hidden = Dense(intermediate_dim, activation='relu')(flat)

	z_mean = Dense(latent_dim)(hidden)
	z_log_var = Dense(latent_dim)(hidden)


	def sampling(args):
	    z_mean, z_log_var = args
	    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
	                              mean=0., stddev=epsilon_std)
	    return z_mean + K.exp(z_log_var) * epsilon

	# note that "output_shape" isn't necessary with the TensorFlow backend
	# so you could write `Lambda(sampling)([z_mean, z_log_var])`
	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

	# we instantiate these layers separately so as to reuse them later
	decoder_hid = Dense(intermediate_dim, activation='relu')
	decoder_upsample = Dense(filters * 14 * 14, activation='relu')

	if K.image_data_format() == 'channels_first':
	    output_shape = (batch_size, filters, 14, 14)
	else:
	    output_shape = (batch_size, 14, 14, filters)

	decoder_reshape = Reshape(output_shape[1:])
	decoder_deconv_1 = Conv2DTranspose(filters,
	                                   kernel_size=num_conv,
	                                   padding='same',
	                                   strides=1,
	                                   activation='relu')
	decoder_deconv_2 = Conv2DTranspose(filters,
	                                   kernel_size=num_conv,
	                                   padding='same',
	                                   strides=1,
	                                   activation='relu')
	if K.image_data_format() == 'channels_first':
	    output_shape = (batch_size, filters, 29, 29)
	else:
	    output_shape = (batch_size, 29, 29, filters)
	decoder_deconv_3_upsamp = Conv2DTranspose(filters,
	                                          kernel_size=(3, 3),
	                                          strides=(2, 2),
	                                          padding='valid',
	                                          activation='relu')
	decoder_mean_squash = Conv2D(img_chns,
	                             kernel_size=2,
	                             padding='valid',
	                             activation='sigmoid')

	hid_decoded = decoder_hid(z)
	up_decoded = decoder_upsample(hid_decoded)
	reshape_decoded = decoder_reshape(up_decoded)
	deconv_1_decoded = decoder_deconv_1(reshape_decoded)
	deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
	x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
	x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

	# instantiate VAE model
	vae = Model(x, x_decoded_mean_squash)

	# Compute VAE loss
	xent_loss = img_rows * img_cols * metrics.binary_crossentropy(
	    K.flatten(x),
	    K.flatten(x_decoded_mean_squash))
	kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	vae_loss = K.mean(xent_loss + kl_loss)
	vae.add_loss(vae_loss)

	vae.compile(optimizer='rmsprop')
	#vae.summary()

	# build a digit generator that can sample from the learned distribution
	decoder_input = Input(shape=(latent_dim,))
	_hid_decoded = decoder_hid(decoder_input)
	_up_decoded = decoder_upsample(_hid_decoded)
	_reshape_decoded = decoder_reshape(_up_decoded)
	_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
	_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
	_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
	_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
	generator = Model(decoder_input, _x_decoded_mean_squash)

	return vae, generator


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