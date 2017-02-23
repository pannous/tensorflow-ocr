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
import letter
# learning_rate = 0.03 # divergence even on overfit
# learning_rate = 0.003 # quicker overfit
learning_rate = 0.0003

training_steps = 500000
batch_size = 64
size = text.canvas_size


data_format={
	input_width: size,
	output_width: 2,  # x,y position
	# output_width: 4,  # x,y start+end position (box)
}
def denseConv(net):
	# type: (layer.net) -> None
	print("Building dense-net")
	net.reshape(shape=[-1, size, size, letter.color_channels])  # Reshape input picture
	net.buildDenseConv(nBlocks=1)
	# net.rnn(text.max_word_length)
	net.regression(dimensions=2) # for


""" Baseline tests to see that your model doesn't have any bugs and can learn small test sites without efforts """

# net = layer.net(layer.baseline, input_width=size, output_width=nClasses, learning_rate=learning_rate)
# net.train(data=data, test_step=1000)  # run

""" here comes the real network """

net = layer.net(denseConv, input_width=size, output_width=2, learning_rate=learning_rate)

# net.train(data=data,steps=50000,dropout=0.6,display_step=1,test_step=1) # debug
# net.train(data=data, steps=training_steps,dropout=0.6,display_step=5,test_step=20) # test
net.train(data=data, dropout=.6, display_step=1, test_step=10) # run resume

# net.predict() # nil=random
# net.generate(3)  # nil=random
