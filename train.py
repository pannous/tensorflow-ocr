#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This example uses a convolutional stack followed by a recurrent stack
and a CTC logloss function to perform optical character recognition
of generated text images. I have no evidence of whether it actually
learns general shapes of text, or just is able to recognize all
the different fonts thrown at it...the purpose is more to demonstrate CTC
inside of Keras.  Note that the font list may need to be updated
for the particular OS in use.

This starts off with 4 letter words.  For the first 12 epochs, the
difficulty is gradually increased using the TextImageGenerator class
which is both a generator class for test/train data and a Keras
callback class. After 20 epochs, longer sequences are thrown at it
by recompiling the model to handle a wider image and rebuilding
the word list to include two words separated by a space.

Based on a script by Mike Henry, with modifications to the model and training procedure
'''
import os
import itertools
import codecs
import re
import datetime
from random import random, randint,uniform

import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD, Adam
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import easygui

MODEL_DIR = 'weights'

# chars  = u'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
chars = u'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß0123456789!@#$%^&*()[]{}-_=+\\|"\'`;:/.,?><~ '
pattern = r'^[A-Za-z0-9 ]+$'



np.random.seed(55)

# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1
def speckle(img):
    # severity = np.random.uniform(-0.2,0.6)
    severity = np.random.uniform(-0.2,0.4)
    gray = uniform(-0.5,0.2) # >0 = whiten!
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur + gray)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise

def paint_text(text, w, h, rotate=False, move=False, multi_fonts=False, background=False):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    with cairo.Context(surface) as context:
        context.set_source_rgb(1, 1, 1)  # White
        context.paint()
        if multi_fonts:
            # Calibri Century Comic Sans  Courier New Futura Georgia
            fonts = ['Century Schoolbook', 'Courier', 'Arial', 'STIX','Tahoma','Times New Roman','Trebuchet MS',
                     'Verdana','Wide Latin','Calibri','Century','Comic Sans','Courier','New Futura','Georgia',
                     'Lucida','Lucida Console','Magneto','Mistral','URW Chancery L', 'FreeMono','DejaVue Sans Mono']
            font_slant = np.random.choice([cairo.FONT_SLANT_NORMAL,cairo.FONT_SLANT_ITALIC,cairo.FONT_SLANT_OBLIQUE])
            font_weight = np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL, cairo.FONT_WEIGHT_NORMAL])
            context.select_font_face(np.random.choice(fonts), font_slant, font_weight)
        else:
            context.select_font_face('Courier', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        # context.set_font_size(25)
        font_size = randint(12, 42)
        context.set_font_size(font_size)
        box = context.text_extents(text)
        border_w_h = (font_size/2, font_size/2)
        # if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
        #     raise IOError('Could not fit string into image. Max char count is too large for given image width.')

        # teach the RNN translational invariance by
        # fitting text box randomly on canvas, with some room to rotate
        min_x = 0 #font_size/4
        min_y = 0# font_size/4
        max_shift_x = w - box[2] - border_w_h[0]
        max_shift_y = h - box[3] - border_w_h[1]

        if max_shift_x <= min_x :
            top_left_x = 10
        else:
            top_left_x = np.random.randint(min_x, int(max_shift_x))

        if move and max_shift_y > min_y + 1:
            top_left_y = np.random.randint(min_y, int(max_shift_y))
        else:
            top_left_y = h // 2
        context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
        context.set_source_rgb(0, 0, 0)
        if text:
            context.show_text(text)

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    a.shape = (h, w, 4)
    a = a[:, :, 0]  # grab single channel
    a = a.astype(np.float32) / 255
    a = np.expand_dims(a, 0)
    if rotate:
        angle = randint(0, 3)
        a = image.random_rotation(a, angle * (w - top_left_x) / w + 1)
        sheer = randint(0, 3)
        a = image.random_shear(a,sheer)
    if background:
        a = speckle(a)

    return a


def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret


# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        # ord(char)
        ret.append(chars.find(char))
    return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        # ret += chr(c)
        if c == len(chars):  # CTC Blank
            ret.append("")
        else:
            ret.append(chars[c])
    return "".join(ret)


# only a-z and space..probably not too difficult
# to expand to uppercase and symbols

def is_valid_str(in_str):
    search = re.compile(pattern, re.UNICODE).search
    return bool(search(in_str))


# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

def random_word(max_string_len):
    s=""
    l = len(chars)
    for i in range(0,randint(4,max_string_len)):
        s+=chars[randint(0, l - 1)]
    return s

class WtfException(Exception):
    pass

class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, monogram_file, bigram_file, minibatch_size,
                 img_w, img_h, downsample_factor, val_split,
                 absolute_max_string_len=16):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.monogram_file = monogram_file
        self.bigram_file = bigram_file
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

    def get_output_size(self):
        return len(chars) + 1

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words, max_string_len=None, mono_fraction=0.5):
        assert max_string_len <= self.absolute_max_string_len
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words
        self.string_list = [''] * self.num_words
        tmp_string_list = []
        self.max_string_len = max_string_len
        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.Y_len = [0] * self.num_words

        if mono_fraction <1 :
            mono_fraction = 0.2
            random_fraction = 0.3
            for i in range(0,int(self.num_words * random_fraction)):
                word = random_word(max_string_len)
                tmp_string_list.append(word)



        # monogram file is sorted by frequency in english speech
        moo=0
        with codecs.open(self.monogram_file, mode='r', encoding='utf-8') as f:
            for line in f:
                if moo == int(self.num_words * mono_fraction):
                    break
                word = line.rstrip()
                if max_string_len == -1 or max_string_len is None or len(word) <= max_string_len:
                    tmp_string_list.append(word)
                    moo += 1

        # bigram file contains common word pairings in english speech
        with codecs.open(self.bigram_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            l = len(lines)
            for line in lines:
                if len(tmp_string_list) == self.num_words:
                    break
                columns = line.lower().split()
                word = columns[0] + ' ' + columns[1]
                if is_valid_str(word) and \
                        (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
                    tmp_string_list.append(word)
        if len(tmp_string_list) != self.num_words:
            print(len(tmp_string_list) , self.num_words)
            raise IOError('Could not pull enough words from supplied monogram and bigram files. ')
        # interlace to mix up the easy and hard words
        self.string_list[::2] = tmp_string_list[:self.num_words // 2]
        self.string_list[1::2] = tmp_string_list[self.num_words // 2:]

        for i, word in enumerate(self.string_list):
            self.Y_len[i] = len(word)
            self.Y_data[i, 0:len(word)] = text_to_labels(word)
            self.X_text.append(word)
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

        self.cur_val_index = self.val_split
        self.cur_train_index = 0

    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 4:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
                labels[i, 0] = self.blank_label
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = 1
                source_str.append('')
            else:
                len1 = len(self.X_text)
                if len1 > (index + i):
                    a_text = self.X_text[index + i]
                    lable = self.Y_data[index + i]
                else:
                    raise WtfException()
                    print("error")
                    a_text = "error" # how / what now??
                    lable = np.ones([self.absolute_max_string_len]) * -1
                    lable[0:len(a_text)]=text_to_labels(a_text)
                func = self.paint_func(a_text)
                text = func[0, :, :].T
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = text
                else:
                    X_data[i, 0:self.img_w, :, 0] = text
                labels[i, :] = lable
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = self.Y_len[index + i]
                source_str.append(a_text)
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            try:
                ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
                self.cur_train_index += self.minibatch_size
                if self.cur_train_index >= self.val_split:
                    self.cur_train_index = self.cur_train_index % 32
                    (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                        [self.X_text, self.Y_data, self.Y_len], self.val_split)
                yield ret
            except WtfException:
                pass # just try new batch
            except Exception as e:
                print(e)
                # raise
                pass

    def next_val(self):
        while 1:
            try:
                ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
                self.cur_val_index += self.minibatch_size
                if self.cur_val_index >= self.num_words:
                    self.cur_val_index = self.val_split + self.cur_val_index % 32
                yield ret
            except Exception as e:
                print(e)
                # raise
                pass # text don't fit etc

    def on_train_begin(self, logs={}):
        self.build_word_list(16000, 4, 1)
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                  rotate=False, move=False, multi_fonts=False)

    def on_epoch_begin(self, epoch, logs={}):
        # rebind the paint function to implement curriculum learning
        if 3 <= epoch < 6:
            self.paint_func = lambda text: \
                paint_text(text, self.img_w, self.img_h, rotate=False, move=True, multi_fonts=False)
        elif 6 <= epoch < 9:
            self.paint_func = lambda text: \
                paint_text(text, self.img_w, self.img_h, rotate=False, move=True, multi_fonts=True)
        elif epoch >= 9:
            self.paint_func = lambda text: \
            paint_text(text, self.img_w, self.img_h, move=True, multi_fonts=True) # clean
            # hardest :
            # paint_text(text, self.img_w, self.img_h, rotate=True, move=True, multi_fonts=True,background=True)

        if epoch >= 21 and self.max_string_len < 12:
            self.build_word_list(32000, 12, 0.5)


# the actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# This could be beam search with a dictionary and language model.
def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret


class VizCallback(keras.callbacks.Callback):

    def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func
        self.output_dir = os.path.join(MODEL_DIR, run_name)
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(num_proc):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.output_dir, 'weights%03d.h5' % (epoch+1)))
        self.show_edit_distance(256)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(self.num_display_words):
            pylab.subplot(self.num_display_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            pylab.imshow(the_input.T, cmap='Greys_r')
            pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        try:
            pylab.savefig(os.path.join(self.output_dir, 'e%03d.png' % (epoch+1)))
            pylab.close()
        except:
            print("CANT SAVE")
            pass

global first
first=10 # quick eval in first n epochs


def train(run_name, start_epoch, stop_epoch, img_w):
    img_h = 64
    minibatch_size = 16
    global first
    # Input Parameters
    if first>0 and start_epoch<300:
        first-=1
        words_per_epoch = 1600 # debug first
    else:
        words_per_epoch = int(1000*minibatch_size/2)
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))

    # Network parameters
    conv_filters = 16 # * 2 can't relearn !?
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32 *2 # 2 makes it WORSE!
    rnn_size = 512 * 2 # 2 helps a lot

    # minibatch_size = 32

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    fdir = os.path.dirname(get_file('wordlists.tgz',
                                    origin='http://www.mythic-ai.com/datasets/wordlists.tgz', untar=True))

    img_gen = TextImageGenerator(monogram_file=os.path.join(fdir, 'wordlist_mono_clean.txt'),
                                 bigram_file=os.path.join(fdir, 'wordlist_bi_clean.txt'),
                                 minibatch_size=minibatch_size,
                                  img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=(pool_size ** 2),
                                 val_split=words_per_epoch - val_words
                                 )
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    # inner = Conv2D(conv_filters, kernel_size, padding='same',
    #                activation=act, kernel_initializer='he_normal',
    #                name='conv3')(inner)
    # inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max3')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dropout(rate=0.2, name='dropout_dense1a')(inner)
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
    inner = Dropout(rate=0.2, name='dropout_dense1b')(inner)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, dropout=0.3, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, dropout=0.3, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    dense2 = Dense(img_gen.get_output_size(), kernel_initializer='he_normal', name='dense2')
    inner = dense2(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    model0=Model(inputs=input_data, outputs=y_pred)
    model0.summary()
    model0.save(os.path.join(MODEL_DIR, 'model%03d.h5' % (start_epoch + 1)))

# training extension:
    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    if start_epoch==0:
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5) # quick
    elif start_epoch<100:
        sgd = SGD(lr=0.008, decay=1e-5, momentum=0.8, nesterov=True, clipnorm=5) # medium speed
    elif start_epoch>250 and start_epoch<300:
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5) # quick relearn
    else:
        # sgd = SGD(lr=0.00005, decay=1e-4, momentum=0.7, nesterov=True, clipnorm=5) # slow
        # sgd = SGD(lr=0.001, decay=1e-5 , momentum=0.8, nesterov=True, clipnorm=5)  # medium speed
        # sgd = SGD(lr=0.03, decay=1e-5, momentum=0.8, nesterov=True, clipnorm=5)  # high speed relearn
        sgd=Adam(lr=0.00005,decay=1e-5)
        # sgd=Adam(lr=0.1)# to start
        # sgd = Adam()

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # for l in model.layers:
    #     if not "conv" in l.name and not "dense1" in l.name:
    #         l.trainable=False

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    if start_epoch > 0:
        weight_file = os.path.join(MODEL_DIR, os.path.join(run_name, 'weights%02d.h5' % start_epoch ))
        model.load_weights(weight_file,by_name=True,reshape=True) # reshape=True, => FILL!
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    viz_cb = VizCallback(run_name, test_func, img_gen.next_val())

    model.fit_generator(generator=img_gen.next_train(),
                        steps_per_epoch=(words_per_epoch - val_words) // minibatch_size,
                        epochs=stop_epoch,
                        validation_data=img_gen.next_val(),
                        validation_steps=val_words // minibatch_size,
                        callbacks=[viz_cb, img_gen],
                        initial_epoch=start_epoch)

def last_epoch():
    maxi=0
    for date in os.listdir(MODEL_DIR):
        if not os.path.isdir(date): continue
        for f in os.listdir(MODEL_DIR+"/"+date):
            if not f.startswith("weights"): continue
            if len(f)==12:
                i = int(f[7:9])
            else:
                i = int(f[7:10])
            if i>maxi:
                maxi=i
    print("start from last_epoch:",maxi)
    return maxi

def beep(e):
    print('\a')
    print(e)
    easygui.msgbox(e, title="ERROR")

if __name__ == '__main__':
    try:
        run_name = 'last' #datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')

        start_epoch = last_epoch() # 553 good but no sigils / 723 not so good with sigils
        if start_epoch<20:
            train(run_name, start_epoch, 20, 128)
            # increase to wider images and start at epoch 20.
            # The learned weights are reloaded
            train(run_name, 20, 125, 512)
        else:
            train(run_name, start_epoch,10000, 512) # quick eval
            # train(run_name, start_epoch, start_epoch+20, 512) # quick eval
            # train(run_name, start_epoch+20, 10000, 512)
    except Exception as e:
        print(e)
        raise

