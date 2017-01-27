#!/usr/bin/python

import layer
import letter

size = letter.max_size
def denseConv(net):
	# type: (layer.net) -> None
	print("Building dense-net")
	net.reshape(shape=[-1, size, size, 1])  # Reshape input picture
	net.buildDenseConv(nBlocks=1)
	net.classifier() # 10 classes auto

# net=layer.net(alex,input_width=28, output_width=nClasses, learning_rate=learning_rate) # NOPE!?
net = layer.net(denseConv, input_width=size, output_width=letter.nClasses)

net.predict() # nil=random
# net.generate(3)  # nil=random
