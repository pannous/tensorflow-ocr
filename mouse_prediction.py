#!/usr/bin/env python
import sys

try:
	import Tkinter  # Tkinter could be supported by all systems
	app = Tkinter.Tk() #  must be declared before Mat
except Exception as ex:
	print("%s" % ex)
	print(" PLEASE DO:")
	print(" sudo apt-get install python-tk  # for Tkinter ")

import tensorflow as tf
import numpy
import numpy as np
import pyscreenshot as ImageGrab
# import cv2

from os import system

import letter

import matplotlib.pyplot as plt
plt.matshow([[1, 0], [0, 1]], fignum=1)
# print(dir(plt))
# help(plt)
# ax.patch.set_facecolor('None') or ax.patch.set_visible(False).
plt.draw()
system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

i = 0
width = 256
height = 256
size = letter.max_size


def get_mouse_position():
	if sys.platform == 'Windows':
			import win32api
			x, y = win32api.GetCursorPos()
	else:
		x, y = app.winfo_pointerxy()
	return x,y


get_mouse_position()
session=tf.Session()


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

#
# # LOAD MODEL!
import layer
net = layer.net(model="denseConv", input_shape=[size, size])
# net = layer.net(model="denseConv", input_shape=[784])
net.predict()#random : debug


def predict(mat):
	try:
		best=net.predict(mat)+letter.offset
		print("interpreted as: %d %s" % (best , chr(best)))
		plt.title("predicted: "+chr(best))
	except Exception as ex:
		print("%s"%ex)
	# plt.waitforbuttonpress(0)
	# plt.close()



if __name__ == "__main__":
	while 1:
		x,y = get_mouse_position()
		# im = ImageGrab.grab() fullscreen
		# image = ImageGrab.grab([x - 60, y - 20, x + 40, y + 20])
		# image = ImageGrab.grab([x - 14, y - 14, x + 14, y + 14])
		# image = ImageGrab.grab([x - 10, y - 10, x + 10, y + 10])
		size = 24  # 24 pycharm font
		w=h=20
		image = ImageGrab.grab([x - w/2, y - w / 2, x + h / 2, y + h / 2])
		# image = ImageGrab.grab([x, y, x + 30, y + 30])
		# help(image.show) Displays this image via preview. This method is mainly intended for debugging purposes
		array = numpy.array(image.getdata()) # (1, 4000, 4)
		mat = array.reshape(image.height, image.width,-1)[:, :, 0]
		if size> image.height:
			mat=numpy.pad(mat, (0,  size- image.height), 'constant', constant_values=255) # 1==white!

		mat = 1 - 2 * mat / 255.  # norm [-1,1] !
		# mat = 1 - mat / 255.  # norm [0,1]! black=1
		# mat = mat / 255.  # norm [0,1]! black=0 (default)

		plt.matshow(mat, fignum=1)
		# plt.imshow(image)
		# print(np.max(mat))
		# print(np.min(mat))
		# print(np.average(mat))
		# convolve(mat)
		predict(mat)

		plt.draw()
		plt.pause(0.01)
		del image
		del mat

		# k = cv2.waitKey(0) & 0xFF  # 0xFF To get the lowest byte.
		# if k == 27: exit(0)


