#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import os
import sys

import numpy as np
from PIL import Image
from keras import backend as K
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import GRU
from keras.models import Model

weight_file = 'best_weights.h5'
alphabet = u'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '

def predict_tensor(images):

  # Model similar to image_ocr.py
  rnn_size = 1024
  dropout = 0.2
  pool_size = 2
  kernel_size = (3, 3)
  time_dense_size = 32
  conv_filters = 16
  img_h = 64
  img_w = 512

  act = 'relu'
  input_data = Input(name='the_input', shape=(img_w, img_h, 1), dtype='float32')
  inner = Conv2D(conv_filters, kernel_size, padding='same', activation=act, name='conv1')(input_data)
  inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
  inner = Conv2D(conv_filters, kernel_size, padding='same', activation=act, name='conv2')(inner)
  inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

  conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
  inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

  inner = Dropout(rate=dropout, name='dropout_dense1a')(inner)
  inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
  inner = Dropout(rate=dropout, name='dropout_dense1b')(inner)

  # Two layers of bidirectional GRUs
  gru_1 = GRU(rnn_size, return_sequences=True, dropout=0.3, name='gru1')(inner)
  gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, name='gru1_b')(inner)
  gru1_merged = add([gru_1, gru_1b])
  gru_2 = GRU(rnn_size, return_sequences=True, dropout=0.3, name='gru2')(gru1_merged)
  gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_merged)

  dense2 = Dense(len(alphabet) + 1, name='dense2')
  inner = dense2(concatenate([gru_2, gru_2b]))
  y_pred = Activation('softmax', name='softmax')(inner)
  model = Model(inputs=input_data, outputs=y_pred)
  model.summary()

  model.load_weights(weight_file, reshape=True, by_name=True)

  prediction = model.predict(images, batch_size=1)
  result = decode_results(prediction)
  return result


def decode_labels(labels):
  ret = []
  for c in labels:
    # ret += chr(c)
    if c == len(alphabet):
      ret.append("")
    else:
      ret.append(alphabet[c])
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
    test_image = "test_image2.png"
  image = Image.open(test_image)
  # image = image.transpose(Image.FLIP_TOP_BOTTOM)
  tensor = np.array(image) / 255.0  # RGBA: h*w*4
  print(tensor.shape)
  # tensor=cv2.resize(tensor,(64,512))
  tensor = tensor.transpose(2, 1, 0)  # 4*w*h
  tensor = tensor[:, :, :, np.newaxis]
  words = predict_tensor([tensor])
  print(words)
