import sys
import Tkinter  # Tkinter could be supported by all systems
import tensorflow as tf
import numpy
import pyscreenshot as ImageGrab

from os import system
app = Tkinter.Tk()
import matplotlib.pyplot as plt
plt.matshow([[1, 0], [0, 1]], fignum=1)
print(dir(plt))
# help(plt)
# ax.patch.set_facecolor('None') or ax.patch.set_visible(False).
plt.draw()
system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

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

['_Image__transformer', '_PngImageFile__idat', '__array_interface__', '__class__', '__copy__', '__delattr__',
 '__dict__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__getattribute__', '__getstate__', '__hash__',
 '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
 '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_copy', '_dump', '_expand', '_makeself',
 '_new', '_open', '_repr_png_', 'category', 'close', 'convert', 'copy', 'crop', 'decoderconfig', 'decodermaxblock',
 'draft', 'effect_spread', 'filename', 'filter', 'format', 'format_description', 'fp', 'frombytes', 'fromstring',
 'getbands', 'getbbox', 'getcolors', 'getdata', 'getextrema', 'getim', 'getpalette', 'getpixel', 'getprojection',
 'height', 'histogram', 'im', 'info', 'load', 'load_end', 'load_prepare', 'load_read', 'mode', 'offset', 'palette',
 'paste', 'png', 'point', 'putalpha', 'putdata', 'putpalette', 'putpixel', 'pyaccess', 'quantize', 'readonly', 'resize',
 'rotate', 'save', 'seek', 'show', 'size', 'split', 'tell', 'text', 'thumbnail', 'tile', 'tobitmap', 'tobytes',
 'toqimage', 'toqpixmap', 'tostring', 'transform', 'transpose', 'verify', 'width']

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
		plt.pause(0.01)

