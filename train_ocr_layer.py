#!/usr/bin/env python
#!/usr/bin/python
import layer
import letter
# import tensorflow as tf
# import layer.baselines

# layer.clear_tensorboard() # Get rid of old runs

data = letter.batch()
input_width, output_width=data.shape[0],data.shape[1]

# learning_rate = 0.03 # divergence even on overfit
learning_rate = 0.003 # quicker overfit
# learning_rate = 0.0003

nClasses =letter.nLetters
training_steps = 500000
batch_size = 64
size = letter.max_size


# OH, it does converge on mnist
def denseConv(net):
	# type: (layer.net) -> None
	print("Building dense-net")
	net.reshape(shape=[-1, size, size, 1])  # Reshape input picture
	net.buildDenseConv(nBlocks=1)
	net.classifier() # 10 classes auto


""" Baseline tests to see that your model doesn't have any bugs and can learn small test sites without efforts """

# net = layer.net(layer.baseline, input_width=size, output_width=nClasses, learning_rate=learning_rate)
# learning_rate: 0.003: full overfit at Step 800
# learning_rate: 0.0003: full overfit at Step 2400

# net = layer.net(layer.baselineDeep3, input_width=size, output_width=nClasses, learning_rate=learning_rate)
# learning_rate: 0.003: overfit 98% at Step 5000
# learning_rate: 0.0003: full overfit at Step 24000

# net = layer.net(layer.baselineBatchNormDeep, input_width=size, output_width=nClasses, learning_rate=learning_rate)
# learning_rate: 0.003: overfit 98% at Step 3000 ++

# net = layer.net(layer.baselineDenseConv, input_width=size, output_width=nClasses, learning_rate=learning_rate)
# learning_rate: 0.003: overfit 98% at Step 3000 ++

# net.train(data=data, steps=training_steps, dropout=0, display_step=100, test_step=1000)  # run

# alex = broken baseline! lol, how?
# net = layer.net(layer.alex, input_width=size, output_width=nClasses, learning_rate=.001)
# net.train(data=data, steps=training_steps, dropout=0.5, display_step=10, test_step=100)  # Alex likes special

# net=layer.net(alex,input_width=28, output_width=nClasses, learning_rate=learning_rate) # NOPE!?
# net=layer.net(denseConv, input_width=28, output_width=nClasses,learning_rate=learning_rate)


net = layer.net(denseConv, input_width=size, output_width=nClasses, learning_rate=learning_rate)
# net.train(steps=50000,dropout=0.6,display_step=1,test_step=1) # debug
net.train(steps=50000,dropout=0.6,display_step=5,test_step=20) # test
# net.train(data=data, steps=training_iters, dropout=.6, display_step=10, test_step=1000) # run
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
