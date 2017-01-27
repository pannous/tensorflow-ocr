import sys
import Tkinter  # Tkinter could be supported by all systems
import tensorflow as tf
import numpy
import pyscreenshot as ImageGrab

from os import system

import layer

app = Tkinter.Tk()
import matplotlib.pyplot as plt
plt.matshow([[1, 0], [0, 1]], fignum=1)
# print(dir(plt))
# help(plt)
# ax.patch.set_facecolor('None') or ax.patch.set_visible(False).
plt.draw()
system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

net=layer.net(model="denseConv", input_shape=[784])
net.predict()#random : debug
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

if __name__ == "__main__":
	while 1:
		x,y = get_mouse_position()
		# im = ImageGrab.grab() fullscreen
		image = ImageGrab.grab([x-60, y-20, x+40 ,y+20])
		# image = ImageGrab.grab([x, y, x + 30, y + 30])
		# help(image.show) Displays this image via preview. This method is mainly intended for debugging purposes
		array = numpy.array(image.getdata())
		# mat= array.reshape(image.height, image.width,4)[:,:,0]
		# plt.matshow(mat, fignum=1)
		plt.imshow(image)
		plt.draw()
		net.predict(array)
		plt.pause(0.01)

