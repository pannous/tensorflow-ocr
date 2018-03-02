import os
import scipy.misc
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, conv2d_transpose


def pool(X):
	return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def uppool(X):
	height, width = X.get_shape().as_list()[1:3]
	return tf.image.resize_images(X, (height * 2, width * 2))


# download from https://i.imgur.com/ytjR2QF.png
image = scipy.misc.imread("snail256.png").astype(np.float32) / 255.0


def make_unet(X):
	depths = [16, 32, 64, 128, 256, 512]
	# TODO figure out how to make batchnorm work

	activation_fn = tf.nn.tanh
	# convolve and half image size a few times
	for depth in depths:
		# 		X = convolution(X, kernel_size=depth, stride=3, activation_fn=activation_fn)
		X = conv2d(X, depth, 3, activation_fn=activation_fn)
		X = pool(X)

	X = conv2d(X, depth, 3, activation_fn=activation_fn)

	# upconcolve and double image size a few times
	for depth in reversed(depths):
		X = uppool(X)
		X = conv2d_transpose(X, depth, 3, activation_fn=activation_fn)

	X = conv2d(X, 3, 3, activation_fn=None)

	return X


input = tf.constant(image.reshape((1, 256, 256, 3)))

output = make_unet(input)

loss = tf.reduce_mean(tf.square(input - output))

# TODO L-BFGS-B should be faster here, but could not get it to work
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

if not os.path.exists("frames"):
	os.mkdir("frames")

def save(frame):
		scipy.misc.imsave("frames/%d.png" % frame, sess.run(output).reshape((256, 256, 3)).clip(0, 1))

for i in range(10000):
	print("\r#" + str(i), end='', flush=True)
	sess.run(train_op)
	if not i % 100:
		save(i/100)
