#!/usr/bin/python
import text
# import letter

# layer.clear_tensorboard() # Get rid of old runs
import numpy as np

data = text.batch(text.Target.position, batch_size=10)
input_width, output_width=data.shape[0],data.shape[1]
# print(data.train.images[0].shape)
# x,y = next(data)
# print(np.array(x).shape)
# print(np.array(y).shape)
# # exit(0)

import layer
import tensorflow as tf
import letter
# learning_rate = 0.03 # divergence even on overfit
# learning_rate = 0.003 # quicker overfit
learning_rate = 0.0003

training_steps = 500000
# batch_size = 64
batch_size = 10
size = text.canvas_size


data_format={
	input_width: size,
	output_width: 2,  # x,y position
	# output_width: 4,  # x,y start+end position (box)
}
def startPositionGanglion(net):
	# type: (layer.net) -> None
	print("Building dense-net")
	net.reshape(shape=[-1, size, size, letter.color_channels])  # Reshape input picture
	# net.buildDenseConv(nBlocks=1)
	net.conv2d(20)
	net.conv2d(1) #  hopefully the heat map activation can learn the start position of our word :
	net.max_pool_with_argmax()
	# net.add(tf.nn.max_pool_with_argmax(net.last_layer))

	# mx = np.max(net.last_layer)
	# mx = np.maximum(net.last_layer)
	# xx=tf.sparse_maximum(net.last_layer, net.last_layer)
	# my = tf.max(net.last_layer, 2)
	# mxy = tf.max(net.last_layer, [1,2])
	# xx = tf.argmax(net.last_layer, 1)
	# yy = tf.argmax(net.last_layer, 2)
	# xy = tf.argmax(xx, 1)
	# xy2 = tf.argmax(xx, 2)
	# print(xx)
	# print(xy)
	# print(xy2)
	# x = tf.argmax(xx, 1)[1]
	# y = tf.argmax(yy, 1)[1]
	#
	# x = tf.cast(x,tf.float32)
	# y = tf.cast(y, tf.float32)
	# # z = net.dense(1)
	# c = layer.tf.concat([x,y],axis=0)
	# net.add(c)
	# net.rnn(text.max_word_length)
	net.regression(dimensions=2) # for


def denseConv(net):
	# type: (layer.net) -> None
	print("Building dense-net")
	net.reshape(shape=[-1, size, size, letter.color_channels])  # Reshape input picture
	net.buildDenseConv(nBlocks=1)


""" Baseline tests to see that your model doesn't have any bugs and can learn small test sites without efforts """

# net = layer.net(layer.baseline, input_width=size, output_width=nClasses, learning_rate=learning_rate)
# net.train(data=data, test_step=1000)  # run

""" here comes the real network """

# net = layer.net(denseConv, input_width=size, output_width=2, learning_rate=learning_rate)
net = layer.net(startPositionGanglion, input_width=size, output_width=2, learning_rate=learning_rate)

# net.train(data=data,steps=50000,dropout=0.6,display_step=1,test_step=1) # debug
# net.train(data=data, steps=training_steps,dropout=0.6,display_step=5,test_step=20) # test
net.train(data=data, dropout=.6, display_step=5, test_step=100) # run resume

# net.predict() # nil=random
# net.generate(3)  # nil=random
