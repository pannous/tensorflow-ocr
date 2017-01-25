#!/usr/bin/env python
#!/usr/bin/python
import tensorflow as tf
import layer
import letter
# layer.clear_tensorboard()
data = letter.batch()
input_width, output_width=data.shape[0],data.shape[1]

learning_rate = 0.0003
nClasses =letter.nLetters
training_iters = 500000
batch_size = 64
size = letter.max_size

def baseline(net):
	# type: (layer.net) -> None
	# net.dense(400, activation=tf.nn.tanh,bn=1)  # Test Accuracy:  0.6875 with batch_norm
	net.dense(400, activation=tf.nn.tanh)  # 	Test; Accuracy:  0.609375
	# net.dense(400,depth=3, activation=tf.nn.sigmoid, bn=1)
  # Step 156000 Loss= 0.871664 Accuracy= 0.700 Time= 1400s 			Test Accuracy:  0.53125 very slow + low

def baselineDeep(net):
	# type: (layer.net) -> None
	# net.batchnorm() # start lower, else no effect
	# net.dense(hidden=200,depth=8,dropout=True) # 50%

	# net.dense(hidden=200, depth=5, bn=True)  # BETTER!!
	# Interesting: the losses dropping, 0.0 accuracy
	# """Step 81000 Loss= 0.956690 Accuracy= 0.000 Time= 1004s 			Test Accuracy:  0.0"""

	# net.dense(hidden=200, depth=5, bn=True)  # BETTER!!
	net.dense(hidden=2000,depth=2,dropout=True)
	# net.dense(200, depth=2, act=tf.nn.tanh)
	# net.dense(400, depth=2, act=tf.nn.tanh)
	# net.denseNet(20, depth=10)
	# net.denseNet(40, depth=4)

	# net.dense(100,dropout=True) # Test Accuracy: 0.960938 LESS IS MORE!
	# net.classifier() # 10 classes auto
	return

# OH, it does converge on mnist
def denseConv(net):
	# type: (layer.net) -> None
	print("Building dense-net")
	# tf.image.crop_and_resize()
	net.reshape(shape=[-1, 28, 28, 1])  # Reshape input picture
	# net.conv([3, 3, 1, 64])
	net.buildDenseConv()
	# net.dense(96*3)
	net.classifier() # 10 classes auto


# OK, not bad, Alex!
def alex(net):
	# type: (layer.net) -> None
	print("Building Alex-net")
	net.reshape(shape=[-1, 28, 28, 1])  # Reshape input picture
	# net.batchnorm()
	net.conv([3, 3, 1, 64])
	net.conv([3, 3, 64, 128])
	net.conv([3, 3, 128, 256])
	net.dense(1024,activation=tf.nn.relu)
	net.dense(1024,activation=tf.nn.relu)


net = layer.net(baseline, input_width=size, output_width=nClasses, learning_rate=learning_rate)
# net = layer.net(baselineDeep, input_width=28, output_width=nClasses, learning_rate=learning_rate)
# net=layer.net(alex,input_width=28, output_width=nClasses, learning_rate=learning_rate) # NOPE!?
# net=layer.net(denseConv, input_width=28, output_width=nClasses,learning_rate=learning_rate)
# net.train(steps=50000,dropout=0.6,display_step=1,test_step=1) # debug
# net.train(steps=50000,dropout=0.6,display_step=5,test_step=20) # test
net.train(data=data, steps=training_iters, dropout=.6, display_step=10, test_step=1000) # run
# net.predict() # nil=random
# net.generate(3)  # nil=random

# refeed projection results as input via partial_run
# a = array_ops.placeholder(dtypes.float32, shape=[])
# b = array_ops.placeholder(dtypes.float32, shape=[])
# c = array_ops.placeholder(dtypes.float32, shape=[])
# r1 = math_ops.add(a, b)
# r2 = math_ops.mul(r1, c)
# h = sess.partial_run_setup([r1, r2], [a, b, c])
# res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
# res = sess.partial_run(h, r2, feed_dict={c: res})
