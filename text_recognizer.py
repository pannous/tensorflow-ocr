#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import itertools
import sys

import numpy as np
from PIL import Image
from keras.models import load_model

# weight_file = 'best_weights.h5'
# weight_file = 'current_weights.h5'
weight_file = 'weights_ascii.h5'

chars = u'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß0123456789!@#$%^&*()[]{}-_=+\\|"\'`;:/.,?><~ '


global model
model=None

def load_model():
  global model
  model = load_model(weight_file)
  # model.load_weights(weight_file, reshape=True, by_name=True)

def predict_tensor(images):
  if not model: load_model()
  prediction = model.predict(images, batch_size=1,verbose=1)
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
  print(tensor.shape)
  # tensor=cv2.resize(tensor,(64,512))
  tensor = tensor.transpose(2, 1, 0)  # 4*w*h
  tensor = tensor[:, :, :, np.newaxis]
  words = predict_tensor([tensor])
  print(words)
