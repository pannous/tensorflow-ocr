#!/usr/bin/env python
import sys

import matplotlib.pyplot as plt
import numpy
import numpy as np
# import gtk
# gtk.set_interactive(False)
import pyscreenshot
import tensorflow as tf

import recognize

try:
	import Tkinter
	app = Tkinter.Tk() #  must be declared before Mat
except Exception as ex:
	import tkinter
	app = tkinter.Tk()  # must be declared before Mat

# import cv2
plt.matshow([[1, 0], [0, 1]], fignum=1)
# print(dir(plt))
# help(plt)
# ax.patch.set_facecolor('None') or ax.patch.set_visible(False).
plt.draw()

# if mac
# system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

i = 0
width = 256
height = 256


def get_mouse_position():
	if sys.platform == 'Windows':
			import win32api
			x, y = win32api.GetCursorPos()
	else:
		x, y = app.winfo_pointerxy()
	return x,y


get_mouse_position()
session=tf.Session()


if __name__ == "__main__":
	while 1:
		x,y = get_mouse_position()
		# im = ImageGrab.grab() fullscreen
		# image = ImageGrab.grab([x - 60, y - 20, x + 40, y + 20])
		# image = ImageGrab.grab([x - 14, y - 14, x + 14, y + 14])
		# image = ImageGrab.grab([x - 10, y - 10, x + 10, y + 10])
		w=512
		h=64
		# image = pyscreenshot.grab([x - w/2, y - h / 2, x + w / 2, y + h / 2])
		image = pyscreenshot.grab([x - 10, y - 10, x + w - 10, y + h - 10])
		# image = pyscreenshot.grab([x, y, x + w, y + h])
		tensor = np.array(image) / 255.0  # RGBA: h*w*4
		print(tensor.shape)
		if len(tensor.shape) == 2:
			tensor = tensor.transpose((1, 0))
			tensor = tensor[np.newaxis, :, :, np.newaxis]
		elif len(tensor.shape)==3:
			tensor = tensor.transpose((2, 1, 0))  # 4*w*h
			tensor = tensor[:, :, :, np.newaxis]
		# tensor=cv2.resize(tensor,(64,512))
		"""

 TEST Text 01234 Hello     <- point your mouse here
 
"""
		# help(image.show) Displays this image via preview. This method is mainly intended for debugging purposes
		array = numpy.array(image.getdata()) # (1, 4000, 4)
		mat = array.reshape(image.height, image.width,-1)[:, :, 0]
		# if size> image.height:
		# 	mat=numpy.pad(mat, (0,  size- image.height), 'constant', constant_values=255) # 1==white!

		# mat = 1 - 2 * mat / 255.  # norm [-1,1] !
		# mat = 1 - mat / 255.  # norm [0,1]! black=1
		mat = mat / 255.  # norm [0,1]! black=0 (default)

		plt.matshow(mat, fignum=1)
		# plt.imshow(image)

		words = recognize.predict_tensor([tensor])
		if len(words)>0:
			best = words[0]
		else:
			best = "???"
		print("interpreted as: %s" % (best))
		plt.title("predicted: " + best)

		plt.draw()
		plt.pause(0.01)
		del image
		del mat

		# k = cv2.waitKey(0) & 0xFF  # 0xFF To get the lowest byte.
		# if k == 27: exit(0)


