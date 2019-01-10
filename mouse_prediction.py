#!/usr/bin/env python
import numpy

import numpy as np
import sys

import matplotlib.pyplot as plt
import pyscreenshot

from text_recognizer import predict_tensor

try:
  import Tkinter as tkinter
except Exception as ex:
  import tkinter

if sys.platform == 'Windows':
  import win32api # GetCursorPos

app = tkinter.Tk()  # must be declared before Mat

plt.matshow([[1, 0], [0, 1]], fignum=1)
plt.draw()

# if mac
# system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

i = 0

def get_mouse_position():
  if sys.platform == 'Windows':
    x, y = win32api.GetCursorPos()
  else:
    x, y = app.winfo_pointerxy()
  return x, y


if __name__ == "__main__":
  while 1:
    x, y = get_mouse_position()
    # im = ImageGrab.grab() fullscreen
    # image = ImageGrab.grab([x - 60, y - 20, x + 40, y + 20])
    # image = ImageGrab.grab([x - 14, y - 14, x + 14, y + 14])
    # image = ImageGrab.grab([x - 10, y - 10, x + 10, y + 10])
    w = 512
    h = 64
    # image = pyscreenshot.grab([x - w/2, y - h / 2, x + w / 2, y + h / 2])
    # image = pyscreenshot.grab([x - 10, y - 10, x + w - 10, y + h - 10]) # pointer
    image = pyscreenshot.grab([x - 10, y , x + w - 10, y + h] ) # cursor
    # image = pyscreenshot.grab([x, y, x + w, y + h])
    mat = np.array(image) / 255.0  # RGBA: h*w*4

    lines=numpy.average(mat, axis=1)
    # todo make model robust to extra text
    argmax = numpy.argmax(lines) # most white
    argmin = numpy.argmin(lines) # most black
    # if(argmax<argmin):
    #   mat[:,:argmax,:]=1. # fill white above
    # if(argmin<argmax):
    #   mat[:,argmax:,:]=1. # fill white below
    # todo: what if invert image!?

    tensor = mat
    print(tensor.shape)
    # tensor=cv2.resize(tensor,(64,512))
    if len(tensor.shape) == 2:
      tensor = tensor.transpose((1, 0))
      tensor = tensor[np.newaxis, :, :, np.newaxis]
    elif len(tensor.shape) == 3:
      mat = numpy.average(tensor, axis=2)  # R+G+B
      tensor = tensor.transpose((2, 1, 0))  # 4*w*h
      tensor = tensor[:, :, :, np.newaxis]

    # mat = 1 - 2 * mat / 255.  # norm [-1,1] !
    # mat = 1 - mat / 255.  # norm [0,1]! black=1
    # mat = mat / 255.  # norm [0,1]! black=0 (default)

    """

 TEST Text 01234 Hello     <- point your mouse here
 
"""

    plt.matshow(mat, fignum=1)
    # plt.imshow(image)

    histogram = numpy.histogram(mat, bins=10, range=None, normed=False, weights=None, density=None)
    print(argmax)

    words = predict_tensor(tensor)
    if len(words) > 0:
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
