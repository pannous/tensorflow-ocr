#!/usr/bin/python
import layer
import letter

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
net = layer.net(model="denseConv", input_shape=[28, 28])
# net = layer.net(model="denseConv", input_shape=[784])
net.predict()  # random : debug

# net.generate(3)  # nil=random

def predict(mat,norm=True):
	try:
		if norm:
			mat = 1 - 2 * mat / 255.  # norm [-1,1] !
			# mat = 1 - mat / 255.  # norm [0,1]!
			# mat = mat / 255.  # norm [0,1]!
		best = net.predict(mat)
		print(chr(best))
	except Exception as ex:
		print("%s" % ex)
	# plt.waitforbuttonpress(0)
	# plt.close()

l= letter.letter()


# noinspection PyTypeChecker
def convolve(mat):
	X = 1
	t = letter.letter(font="Menlo.ttc", size=size, char="X")
	filter = t.matrix()
	filter = 255 - 2 * filter  # black->1 white->-1
	filter = filter / 255.
	mat = 255 - mat  # white->0
	mat = (mat - 128) / 128.  # 1...-1 !
	# plt.matshow(filter,fignum=2)
	mat = np.reshape(mat, [1, image.height, image.width, 1]).astype(np.float32)
	# f_ = np.reshape([[f]], [3, 3])
	filter = np.reshape(filter, [size, size, 1, 1]).astype(np.float32)
	conv = tf.nn.conv2d(mat, filter, strides=[1, 1, 1, 1], padding='SAME')  # VALID
	ret = session.run(conv)
	ret = np.reshape(ret, [size, size])
	plt.matshow(ret, fignum=2)
	score = numpy.max(ret)
	plt.title("score: %f" % score)

