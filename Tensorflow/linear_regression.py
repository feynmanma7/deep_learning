#encoding: utf-8

import tensorflow as tf
import numpy as np

np.random.seed(20170430)

# y = .3 * x1 + .7 * x2 + .4 * x3 + .2 * x4 + .2 + N(0, 0.001)

filename_queue = tf.train.string_input_producer(
	['train_1.csv', 'train_2.csv'])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[tf.constant(1.0, dtype='float64')]] * 5
buf = tf.decode_csv(value, record_defaults, )
X = tf.stack([buf[:-1]])
y = buf[-1]

learning_rate = 5e-3
epochs = 10000

W = tf.Variable(np.random.random((4, 1)), name="weight", dtype='float64')
b = tf.Variable(np.random.rand(), name='bias', dtype='float64')

tf.summary.scalar('weight', W[0][0])
tf.summary.scalar('weight', W[1][0])
tf.summary.scalar('bias', b)

pred = tf.add(tf.reduce_sum(tf.multiply(X, W)), b)
loss = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

tf.summary.scalar('loss', loss)

writer = tf.summary.FileWriter('./graphs', 
	tf.get_default_graph())

init = tf.global_variables_initializer()

merged = tf.summary.merge_all()

with tf.Session() as sess:
	sess.run(init)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	
	for epoch in range(epochs):
		sess.run(optimizer)

		if epoch % 100 == 0 or epoch == epochs - 1:
			print('Epoch %s, loss=%s, W=%s, b=%s' %
				 (epoch, sess.run(loss), sess.run(W), sess.run(b)))

		summary = sess.run(merged)
		writer.add_summary(summary, epoch)

	coord.request_stop()
	coord.join(threads)

writer.close()
