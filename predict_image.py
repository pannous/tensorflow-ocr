#!/usr/bin/python
import layer
import letter
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from letter import letter as l
size = letter.max_size
# def denseConv(net):
# 	# type: (layer.net) -> None
# 	print("Building dense-net")
# 	net.reshape(shape=[-1, size, size, 1])  # Reshape input picture
# 	net.buildDenseConv(nBlocks=1)
# 	net.classifier()  # 10 classes auto
# # net=layer.net(alex,input_width=28, output_width=nClasses, learning_rate=learning_rate) # NOPE!?
# net = layer.net(denseConv, input_width=size, output_width=letter.nClasses)

# # LOAD MODEL!
net = layer.net(model="denseConv", input_shape=[size, size])
# net = layer.net(model="denseConv", input_shape=[784])
# net.predict()  # random : debug

# net.generate(3)  # nil=random
def norm(mat):
	mat = 1 - 2 * mat / 255.  # norm [-1,1] !
# mat = 1 - mat / 255.  # norm [0,1]!
# mat = mat / 255.  # norm [0,1]!

def predict(mat,norm=False):
	try:
		best = net.predict(mat)
		best = chr(best)
		print(best)
		return best
	except Exception as ex:
		print("%s" % ex)
	# plt.waitforbuttonpress(0)
	# plt.close()


# noinspection PyTypeChecker
def convolve(mat):
	X = 1
	session=tf.Session()
	t = letter.letter(font="Menlo.ttc", size=size, char="X")
	mat = t.matrix()
	# plt.matshow(filter,fignum=2)
	# mat = np.reshape(mat, [1, image.height, image.width, 1]).astype(np.float32)
	# f_ = np.reshape([[f]], [3, 3])
	filter = np.reshape(filter, [size, size, 1, 1]).astype(np.float32)
	conv = tf.nn.conv2d(mat, filter, strides=[1, 1, 1, 1], padding='SAME')  # VALID
	ret = session.run(conv)
	ret = np.reshape(ret, [size, size])
	plt.matshow(ret, fignum=2)
	score = np.max(ret)
	plt.title("score: %f" % score)


assert predict(l(char="a").matrix())=="A" # Case insensitive prediction for now
