#!/usr/bin/env python
#!/usr/bin/python
import tensorflow as tf
import layer
import letter

data = letter.batch()

learning_rate = 0.01
# learning_rate = 0.0003
training_iters = 500000
batch_size = 64


# best with lr ~0.001
def baseline(net):
	# type: (layer.net) -> None
	# net.batchnorm() # start lower, else no effect
	# net.dense(hidden=200,depth=8,dropout=True) # 50%
	# net.dense(hidden=200,depth=8,dropout=False) # BETTER!!
	# net.dense(hidden=2000,depth=2,dropout=True)
	# net.dense(400,act=None)#  # ~95% we can do better:
	net.dense(400, activation=tf.nn.tanh)# 0.996 YAY  only 0.985 on full set, Step 5000 flat
	# net.dense(200, depth=2, act=tf.nn.tanh)# 0.98 on full set, Step 5000 flat
	# net.dense(400, depth=2, act=tf.nn.tanh)# 0.98 on full set
	# net.denseNet(20, depth=10)
	# net.denseNet(40, depth=4)

	# net.dense(100,dropout=True) # Test Accuracy: 0.960938 LESS IS MORE!
	# net.classifier() # 10 classes auto
	return # 0.957% without any model!!

# OH, it does converge!!
# Step 12890 Loss= 0.272316 Accuracy= 0.984 			Test Accuracy: 0.9975 WOW, record on mnist!
# Step 4290 Loss= 0.000048 Accuracy= 1.000 			Test Accuracy:  1.0
def denseConv(net):
	# type: (layer.net) -> None
	print("Building dense-net")
	net.reshape(shape=[-1, 28, 28, 1])  # Reshape input picture
	# net.conv([3, 3, 1, 64])
	net.buildDenseConv()
	net.dense(96*3)
	net.classifier() # 10 classes auto


# OK, not bad, Alex!
#  Step 6490 Loss= 0.000908 Accuracy= 1.000                        Test Accuracy: 0.995
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


# net=layer.net(baseline, data, learning_rate=0.001)
# net=layer.net(alex,data, learning_rate=0.001) # NOPE!?
net=layer.net(denseConv,data, learning_rate)

# net.train(steps=50000,dropout=0.6,display_step=1,test_step=1) # debug
# net.train(steps=50000,dropout=0.6,display_step=5,test_step=20) # test
net.train(data=data, steps=training_iters, dropout=.7, display_step=100, test_step=100) # run
# net.predict() # nil=random
# net.generate(3)  # nil=random
