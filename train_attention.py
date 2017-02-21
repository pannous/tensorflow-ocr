#!/usr/bin/python
import tensorflow as tf
import layer
import letter
from letter import Target,batch


# learning_rate = 0.0003
target=Target.position
learning_rate = 0.0001
nClasses = letter.nClasses[target]
size = letter.max_size
training_iters = 500000
batch_size = 64
data = batch(batch_size,target)
print("data.shape %s"% data.shape)

# best with lr ~0.001
def baseline(net):
	# type: (layer.net) -> None
	# net.batchnorm() # start lower, else no effect
	net.dense(400, activation=tf.nn.tanh)
	net.regression(nClasses)# regression
	# net.denseNet(40, depth=4)
	return


# net=layer.net(baseline, input_shape=[28,28], output_width=nClasses,learning_rate=0.001)
# net=layer.net(alex,data, learning_rate=0.001) # NOPE!?
# net=layer.net(denseConv, input_shape=[size, size], output_width=nClasses,learning_rate=learning_rate)

# net.train(steps=50000,dropout=0.6,display_step=1,test_step=1) # debug
# net.train(steps=50000,dropout=0.6,display_step=5,test_step=20) # test
net.train(data=data, steps=training_iters, dropout=.6, display_step=10, test_step=100) # run

# net.predict() # nil=random
# net.generate(3)  # nil=random
