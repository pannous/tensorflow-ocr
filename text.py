# coding=utf-8
from enum import Enum

import PIL.ImageOps
import numpy
from PIL import Image, ImageDraw, ImageFont

import letter
from extensions import *

num_characters = letter.nLetters # ascii: 127
word_file = "/usr/share/dict/words"
WORDS = open(word_file).read().splitlines()

class Target(Enum):  # labels
	word = 1
	text = 2
	box = 3 # start,end
	position = 4
	position_hot = 40
	position_x = 44
	start = 4 #position
	end = 5 #position
	style = 6
	angle = 7
	size = 8



def random_word():
	return pick(WORDS)
	pass # Don't (just) use dictionary because we really want to ocr passwords too



def pos_to_arr(pos):
	return [pos['x'],pos['y']]


max_size = letter.max_size
max_word_length= 15
canvas_size=300 # Arbitrary,  shouldn't need to be specified in advance when doing inference

def pad(vec, pad_to=max_word_length, one_hot=False, terminal_symbol=0):
	for i in range(0, pad_to - len(vec)):
		if one_hot:
			vec.append([terminal_symbol] * num_characters)
		else:
			vec.append(terminal_symbol)
	return vec

class data():
	def __init__(self):
		self.input_shape = None
		self.output_shape = None

	def __next__(self):
		return self.next_batch()

	def __iter__(self):
		# return next(self.generator)
		return self.next_batch()

	def next_batch(self):
		raise Exception("abstract data class must be implemented")


class batch(data):

	def __init__(self, target=Target.word, batch_size=64):
		#super().__init__()
		self.batch_size=batch_size
		self.target= target
		self.shape=[max_size * max_size, max_word_length * letter.nLetters]
		# self.shape=[batch_size,max_size,max_size,len(letters)]
		self.train= self
		self.test = self
		# self.test.images,self.test.labels = self.next_batch() # nonesense!

	def next_batch(self,batch_size=None):
		# type: () -> (list,list)
		words = [word() for i in range(batch_size or self.batch_size)]
		def norm(word):
			# type: (word) -> ndarray
			return word.matrix() # dump  the whole abstract representation as an image
		xs=list(map(norm, words)) # 1...-1 range
		if self.target == Target.word: ys=  [many_hot(w.text, num_characters) for w in words]
		if self.target == Target.size: ys = [l.size for l in words]
		if self.target == Target.position: ys = [pos_to_arr(l.pos) for l in words]
		if self.target == Target.position_x: ys = [l.pos['x'] for l in words]
		if self.target == Target.position_hot:
			ys = [many_hot(pos_to_arr(l.pos), canvas_size, limit=2,swap=True) for l in words]
		return xs, ys


def pick(xs):
	return xs[randint(0,len(xs)-1)]

def many_hot(word, num_classes, offset=0, limit=max_word_length, swap=False):
	labels_many_hot = []
		# for item in items:
	for letter in word:
		item=ord(letter)
		labels_one_hot = numpy.zeros(num_classes)
		if item >= num_classes:
			print("item > num_classes  %s > %d  ignoring" % (item, limit))
		else:
			labels_one_hot[item - offset] = 1
		labels_many_hot.append(labels_one_hot)
		if len(labels_many_hot) > limit:
			print("#items > limit   %s > %d  ignoring rest"%(len(labels_many_hot),limit))
			break
	l = len(labels_many_hot)
	if l < limit:
		pad(labels_many_hot,limit,true)
	if l > limit:
		raise Exception("Too many items: %d > %d"%(l,limit))
	labels_many_hot = np.array(labels_many_hot)
	if swap:
		labels_many_hot = labels_many_hot.swapaxes(0,1)  # .transpose([0,1]) theano.dimshuffle
	# print(labels_many_hot.shape)
	return labels_many_hot


class word():


	def __init__(self, *margs, **args): # optional arguments
		if not args:
			if margs: args=margs[0] # ruby style hash args
			else:args={}
		# super(Argument, self).__init__(*margs, **args)
		# self.name = args['name']		if 'name' in args else None
		# self.family = args['family'] if 'family' in args else pick(families)
		self.font = args['font'] if 'font' in args else pick(letter.fontnames)
		self.size = args['size'] if 'size' in args else pick(letter.sizes)
		self.color= args['color'] if 'color' in args else 'black'#'white'#self.random_color() #  #None #pick(range(-90, 180))
		self.back = args['back'] if 'back' in args else letter.random_color()
		self.angle= args['angle'] if 'angle' in args else 0 #pick(range(-max_angle,max_angle))
		self.pos	= args['pos'] if 'pos' in args else {'x':pick(range(0,canvas_size)),'y':pick(range(0, canvas_size))}
		# self.style= args['style'] if 'style' in args else self.get_style(self.font)# pick(styles)
		self.text = args['text'] if 'text' in args else random_word()
		self.invert = args['invert'] if 'invert' in args else pick([-1, 0, 1])
		# if chaotic: # captcha style (or syntax highlighting?)
		# self.letters=[letter.letter(args,char=char) for char in self.word] # almost java style ;)
		# else: one word, one style!

	def projection(self):
		return self.matrix(),self.ord

	def matrix(self, normed=true):
		# type: (bool) -> ndarray
		matrix = np.array(self.image())
		if normed: matrix = matrix / 255.
		if self.invert == -1:
			matrix = 1 - 2 * matrix # -1..1
		elif self.invert:
			matrix = 1 - matrix # 0..1
		return matrix
		# except:
		# 	return np.array(max_size*(max_size+extra_y))

	def image(self):
		ttf_font = self.load_font()
		padding = self.pos
		size = [canvas_size, canvas_size]
		if self.back:
			img = Image.new('RGBA', size, self.back) # background_color
		else:
			img = Image.new('L', size, 'white')	# grey
		draw = ImageDraw.Draw(img)
		draw.text((padding['x'], padding['y']), self.text, font=ttf_font, fill=self.color)
		if self.angle:
			rot = img.rotate(self.angle, expand=1).resize(size)
			if self.back:
				img = Image.new('RGBA', size, self.back)  # background_color
			else:
				img = Image.new('L', size,'#FFF')#FUCK BUG! 'white')#,'grey')  # # grey
			img.paste(PIL.ImageOps.colorize(rot, (0, 0, 0),self.back ) (0, 0), rot)
		return img

	def load_font(self):
		fontPath = self.font if '/' in self.font else letter.fonts_dir + self.font
		try:
			fontPath = fontPath.strip()
			ttf_font = ImageFont.truetype(fontPath, self.size)
		except:
			print("BAD FONT: " + fontPath,self.size)
			# raise
			exit()
			# raise Exception("BAD FONT: " + fontPath,self.size)
		return ttf_font

	def show(self):
		self.image().show()

	def __str__(self):
		format="text{'%s',size=%d,font='%s',position=%s}" # angle=%d,
		return format % (self.text, self.size, self.font, self.pos)

	def __repr__(self):
		return self.__str__()

	def save(self, path):
		self.image().save(path)


# @classmethod	# can access class cls
# def ls(cls, mypath=None):

# @staticmethod	# CAN'T access class
# def ls(mypath):
import matplotlib.pyplot as plt


def show_matrix(mat):
	plt.matshow(mat, fignum=1)
	# plt.imshow(image)
	plt.draw()
	plt.waitforbuttonpress()


def show_image(image):
	plt.imshow(image)
	plt.draw()
	plt.waitforbuttonpress()


if __name__ == "__main__":
	while 1:
		# l = word(text='hello')
		w = word()
		# l.save("letters/letter_%s_%d.png"%(l.char,l.size))
		print(w)
		try:
			# show_matrix(mat)
			image = w.image()
			show_image(image)
			del(image)
			# mat = w.matrix()
			# print(np.average(mat))
			# print(np.max(mat))
			# print(np.min(mat))
		except KeyboardInterrupt:
			exit()
			# return
			break



