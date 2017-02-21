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
# learning_rate = 0.003 # quicker overfit
learning_rate = 0.0003

nClasses =letter.nLetters
training_steps = 500000
batch_size = 64
size = letter.max_size


# OH, it does converge
# Test Accuracy:  ~0.875 Step 1.000.000 52148s
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

# alex = broken baseline! lol, how?
# net = layer.net(layer.alex, input_width=size, output_width=nClasses, learning_rate=.001)

# net.train(data=data, test_step=1000)  # run

""" here comes the real network """

# net=layer.net(alex,input_width=28, output_width=nClasses, learning_rate=learning_rate) # NOPE!?
net = layer.net(denseConv, input_width=size, output_width=nClasses, learning_rate=learning_rate)

# net.train(data=data,steps=50000,dropout=0.6,display_step=1,test_step=1) # debug
# net.train(data=data, steps=training_steps,dropout=0.6,display_step=5,test_step=20) # test
net.train(data=data, dropout=.6, display_step=10, test_step=1000) # run resume

# net.predict() # nil=random
# net.generate(3)  # nil=random
