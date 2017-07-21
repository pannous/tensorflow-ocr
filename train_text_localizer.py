#!/usr/bin/python
import letter
import text
import layer
import numpy as np

# layer.clear_tensorboard() # Get rid of old runs

# data = text.batch(text.Target.position, batch_size=10)
data = text.batch(text.Target.position_hot, batch_size=10)
# input_width = 300  #data.shape[0]
input_shape=[text.canvas_size, text.canvas_size]
output_shape = [text.canvas_size, 2]  # one hot encoding for (x,y) position of text ~ upper right boundary box corner
# output_shape = 2  # (x,y) todo: one hot encoding INSIDE net!

# print(data.train.images[0].shape)
x,y = next(data)
print(np.array(x).shape)
print(np.array(y).shape)
# exit(0)


# learning_rate = 0.03 # divergence even on overfit
# learning_rate = 0.003 # quicker overfit
learning_rate = 0.0003

training_steps = 500000
# batch_size = 64
batch_size = 10
size = text.canvas_size


data_format={
	'input_width': size,
	'output_width': output_shape,  # x,y position
	# output_width: 4,  # x,y start+end position (box)
}


def positionGanglion(net):
	# type: (layer.net) -> None
	print("Building start position detecting ganglion")
	net.input([300,300])
	net.reshape(shape=[-1, size, size, letter.color_channels])  # Reshape input picture
	# net.buildDenseConv(nBlocks=1)
	net.conv2d(20, pool=False)
	net.conv2d(1, pool=False) #  hopefully the heat map activation can learn the start position of our word :
	net.targets([300,2]) # reduce-max per axis
	net.argmax_2D_loss()
	# net.classifier(dim=2)

def positionRegression(net):
	# type: (layer.net) -> None
	print("Building start position detecting ganglion")
	net.reshape(shape=[-1, size, size, letter.color_channels])  # Reshape input picture
	# net.buildDenseConv(nBlocks=1)
	net.conv2d(20)
	net.argmax2d()
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
net = layer.net(positionGanglion, input_width=size, output_width=output_shape, learning_rate=learning_rate)

# net.train(data=data,steps=50000,dropout=0.6,display_step=1,test_step=1) # debug
# net.train(data=data, steps=training_steps,dropout=0.6,display_step=5,test_step=20) # test
net.train(data=data, dropout=.6, display_step=5, test_step=100) # run resume

# net.predict() # nil=random
# net.generate(3)  # nil=random
