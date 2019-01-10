#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import itertools
import sys

import numpy as np
from keras.models import load_model
from PIL import Image # python -m pip install --upgrade Pillow  # WTF

# weight_file = 'best_weights.h5'
# weight_file = 'current_weights.h5'

weight_file = 'weights_ascii.h5' # learned on noisy data
# weight_file = 'weights_ascii_easy.h5' # no freckles
# weight_file = 'weights_ascii_clean.h5' # pure text
# weight_file = None # use model weights

chars = u'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß0123456789!@#$%^&*()[]{}-_=+\\|"\'`;:/.,?><~ '


global model
model=None

def init_model(model_file="current_model.h5"):
  global model
  model = load_model(model_file)
  model.summary()
  model.load_weights(weight_file, reshape=True, by_name=True)

def predict_tensor(tensor):
  if len(tensor.shape) == 2:
    tensor = tensor.transpose((1, 0))
    tensor = tensor[np.newaxis, :, :, np.newaxis]
  elif len(tensor.shape) == 3:
    tensor = tensor.transpose((2, 1, 0))  # 4*w*h
    tensor = tensor[:, :, :, np.newaxis]

  print(tensor.shape)
  if not model: init_model()
  prediction = model.predict([tensor], batch_size=1, verbose=1)
  result = decode_results(prediction)
  return result


def decode_labels(labels):
  ret = []
  for c in labels:
    # ret += chr(c)
    if c == len(chars):
      ret.append("")
    else:
      ret.append(chars[c])
  return "".join(ret)


# could be extended to beam search with a dictionary and language model.
def decode_results(prediction):
  ret = []
  for j in range(prediction.shape[0]):
    out_best = list(np.argmax(prediction[j, 2:], 1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = decode_labels(out_best)
    ret.append(outstr)
  return ret


if __name__ == '__main__':

  np.random.seed(128)
  if (len(sys.argv) > 1):
    test_image = sys.argv[1]
  else:
    test_image = "test_image.png"
  image = Image.open(test_image)
  # image = image.transpose(Image.FLIP_TOP_BOTTOM)
  tensor = np.array(image) / 255.0  # RGBA: h*w*4
  words = predict_tensor(tensor)
  print(words)
