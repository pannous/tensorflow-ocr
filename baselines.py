# import .net as layer.net
# import net as layer.net
import layer
import tensorflow as tf


class identity(layer.net):  # baseline 0 ;)
	def __init__(self):
		pass
	def predict(self, eval_data=None, model=None):
		return eval_data

def baseline(net):
	# type: (layer.net) -> None
	net.dense(400, activation=tf.nn.tanh)


# over-fitting okay: Step 3000 Loss= 0.301323 Accuracy= 1.000 Time= 20s 			Test Accuracy:  1.0"""
def baselineDeep3(net):
	# type: (layer.net) -> None
	net.dense(400, depth=3, activation=tf.nn.tanh)

def baselineWide(net):
	net.dense(hidden=20000, depth=2, dropout=True)


def baselineBatchNorm(net):
	# type: (layer.net) -> None
	net.batchnorm()
	net.dense(400, activation=tf.nn.tanh, bn=1)
	net.batchnorm()


def baselineBatchNormDeep(net):
	# type: (layer.net) -> None
	net.batchnorm()  # start lower, else no effect
	net.dense(400, depth=3, activation=tf.nn.tanh, bn=1)


# Interesting: losses dropping, still 0.0 accuracy! => predicting zeros!?!
size=10
def baselineDenseConv(net):
	# type: (layer.net) -> None
	print("Building dense-net")
	net.reshape(shape=[-1, size, size, 1])  # Reshape input picture
	net.buildDenseConv(nBlocks=1)
	net.classifier()  # 10 classes auto


# Patience! Alex does converge after ~10k steps ... maybe
def alex(net):
	# type: (layer.net) -> None
	print("Building Alex-net")
	net.reshape(shape=[-1, size, size, 1])  # Reshape input picture
	# net.batchnorm()
	net.conv([3, 3, 1, 64])
	net.conv([3, 3, 64, 128])
	net.conv([3, 3, 128, 256])
	net.dense(1024, activation=tf.nn.relu)
	net.dense(1024, activation=tf.nn.relu)
