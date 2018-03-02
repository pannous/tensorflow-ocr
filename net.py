from __future__ import print_function

import time
import numpy as np
import tensorflow as tf # needs tf > 1.0
from tensorflow.contrib.tensorboard.plugins import projector  # for 3d PCA/ t-SNE
from .tensorboard_util import *

print("tf.__version__:%s" % tf.__version__)

start = int(time.time())

gpu = True
# gpu = False
debug = False  # summary.histogram  : 'module' object has no attribute 'histogram' WTF
# debug = True  # histogram_summary ...

# clear_tensorboard()
if debug:
	set_tensorboard_run(auto_increment=True)
	run_tensorboard(restart=False)

visualize_cluster = False  # NOT YET: 'ProjectorConfig' object has no attribute 'embeddings'

weight_divider = 10.
default_learning_rate = 0.001  # mostly overwritten, so ignore it
decay_steps = 100000
decay_size = 0.1
save_step = 10000  # if you don't want to save snapshots, set to 0

checkpoint_dir = "checkpoints"
_cpu = '/cpu:0'
_gpu = '/GPU:0'

if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)


def nop(): return 0


def closest_unitary(A):
	""" Calculate the unitary matrix U that is closest with respect to the operator norm distance to the general matrix A. """
	try:
		import scipy
		V, __, Wh = scipy.linalg.svd(A)
		return np.matrix(V.dot(Wh))
	except:
		return A


class net:
	def __init__(self, model, input_width=0, output_width=0, input_shape=[], name=0, learning_rate=default_learning_rate):
		self.fully_connected = self.dense  # alias
		device = _gpu if gpu else _cpu
		device = None  # auto
		print("Using device ", device)
		with tf.device(device):
			self.session = tf.Session()
			self.model = model
			self.input_shape = input_shape or [input_width, input_width]
			if not input_width: input_width, _ = self.get_data_shape()
			self.input_width = input_width
			self.last_width = self.input_width
			self.last_shape = self.input_shape
			self.output_width = output_width
			self.num_classes = output_width
			# self.batch_size=batch_size
			self.layers = []
			self.learning_rate = learning_rate
			if isinstance(model, str):
				self.name = model
				self.restore()
				return
			self.name = model.__name__
			if input_width == 0:
				raise Exception("Please set input_width or input_shape")
			if output_width == 0:
				raise Exception("Please set number of classes via output_width")
			self.generate_model(model)

	def get_data_shape(self):
		if self.input_shape:
			if len(self.input_shape) == 1: return [self.input_shape[0], 0]
			return self.input_shape[0], self.input_shape[1]
		try:
			return self.data.shape[0], self.data.shape[-1]
		except:
			raise Exception("Data does not have shape")

	def generate_model(self, model, name=''):
		if not model: return self
		with tf.name_scope('state'):
			self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")  # 1 for testing! else 1 - dropout
			self.train_phase = tf.placeholder(tf.bool, name='train_phase')
			with tf.device(_cpu): self.global_step = tf.Variable(0)
			# dont set, feed or increment global_step, tensorflow will do it automatically
		with tf.name_scope('data'):
			if self.input_shape and len(self.input_shape) == 2:
				shape_ = [None, self.input_shape[0], self.input_shape[1]]  # batch:None
				# todo [None, *self.input_shape]
				self.x = x = self.input = tf.placeholder(tf.float32, shape_, name="input_x")
				self.last_shape = x
			elif self.input_width:
				self.x = x = self.input = tf.placeholder(tf.float32, [None, self.input_width], name="input_x")
			else:
				raise Exception("need input_shape or input_width by now")
			self.last_layer = self.x
			tf.add_to_collection('inputs', self.x)
			self.y = y = self.target = tf.placeholder(tf.float32, [None, self.output_width], name="target_y")
			tf.add_to_collection('targets', self.target)
		with tf.name_scope('model'):
			model(self)
		if (self.last_width != self.output_width):
			self.classifier()  # 10 classes auto

	def dropout(self, keep_rate=0.6):
		self.add(tf.nn.dropout(self.last_layer, keep_rate))

	def add(self, layer):
		self.layers.append(layer)
		self.last_layer = layer
		self.last_shape = layer.get_shape()

	def reshape(self, shape):
		self.last_layer = tf.reshape(self.last_layer, shape)
		self.last_shape = shape
		self.last_width = shape[-1]

	# BN also serve as a stochastic regularizer and makes dropout regularization redundant! Furthermore dropout never really helped when inserted between convolution layers and was most useful between fully connected layers.
	# when applying batchnorm you can drop biases [redundant: BN(x)=ax+b] and must increase learning rate!! ++
	def batchnorm(self, input=None, center=False):  # for conv2d and fully_connected [only!?]
		if input is None: input = self.last_layer
		from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
		with tf.name_scope('batchnorm') as scope:
			# mean, var = tf.nn.moments(input, axes=[0, 1, 2])
			# self.batch_norm = tf.nn.batch_normalization(input, mean, var, offset=1, scale=1, variance_epsilon=1e-6)
			# self.last_layer=self.batch_norm
			# activation_fn all in one go!  sigmoid: center=true! relu:center=False?
			# is_training why not automatic??  bad implementation: placeholder -> needs_moments
			# vs low level nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)   # nn.fused_batch_norm
			# data_format: A string. `NHWC` vs NCHW WHY NOT AUTO??
			# activation_fn inline
			train_op = batch_norm(input, is_training=True, center=center, updates_collections=None, scope=scope)
			test_op = batch_norm(input, is_training=False, updates_collections=None, center=False, scope=scope, reuse=True)
			output = tf.cond(self.train_phase, lambda: train_op, lambda: test_op)
			# output=self.debug_print(output)
			self.add(output)
			return output

	def addLayer(self, nChannels, nOutChannels, do_dropout):
		ident = self.last_layer
		self.batchnorm()
		# self.add(tf.nn.relu(ident)) # nChannels ?
		self.conv([3, 3, nChannels, nOutChannels], pool=False, dropout=do_dropout, norm=tf.nn.relu)  # None
		concat = tf.concat(axis=3, values=[ident, self.last_layer])
		print("concat ", concat.get_shape())
		self.add(concat)

	def addTransition(self, nChannels, nOutChannels, do_dropout):
		self.batchnorm()
		self.add(tf.nn.relu(self.last_layer))
		self.conv([1, 1, nChannels, nOutChannels], pool=True, dropout=do_dropout, norm=None)  # pool (2, 2)

	# self.add(tf.nn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))


	# Fully connected 'pyramid' layer, allows very high learning_rate >0.1 (but don't abuse)
	# NOT TO BE CONFUSED with buildDenseConv below!
	def fullDenseNet(self, hidden=20, depth=3, act=tf.nn.tanh, dropout=True, norm=None):  #
		if hidden > 100: print("WARNING: denseNet uses O(n^2) quadratic memory for " + str(hidden)) + " hidden units"
		if depth < 3: print(
			"WARNING: did you mean to use Fully connected layer 'dense'? Expecting depth>3 vs " + str(depth))
		inputs = self.last_layer
		inputs_width = self.last_width
		width = hidden
		while depth > 0:
			with tf.name_scope('DenNet_{:d}'.format(width)) as scope:
				print("dense width ", inputs_width, "x", width)
				nr = len(self.layers)
				weights = tf.Variable(tf.random_uniform([inputs_width, width], minval=-1. / width, maxval=1. / width),
				                      name="weights")
				bias = tf.Variable(tf.random_uniform([width], minval=-1. / width, maxval=1. / width),
				                   name="bias")  # auto nr + context
				dense1 = tf.matmul(inputs, weights, name='dense_' + str(nr)) + bias
				tf.summary.histogram('dense_' + str(nr), dense1)
				tf.summary.histogram('dense_' + str(nr) + '/sparsity', tf.nn.zero_fraction(dense1))
				tf.summary.histogram('weights_' + str(nr), weights)
				tf.summary.histogram('weights_' + str(nr) + '/sparsity', tf.nn.zero_fraction(weights))
				tf.summary.histogram('bias_' + str(nr), bias)

				if act: dense1 = act(dense1)
				if norm: dense1 = self.norm(dense1, lsize=1)  # SHAPE!
				if dropout: dense1 = tf.nn.dropout(dense1, self.keep_prob)
				self.add(dense1)
				self.last_width = width
				inputs = tf.concat(axis=1, values=[inputs, dense1])
				inputs_width += width
				depth = depth - 1
		self.last_width = width

	# Densely Connected Convolutional Networks https://arxiv.org/abs/1608.06993
	def buildDenseConv(self, nBlocks=3, nChannels=64, magic_factor=0):
		if magic_factor: print("magic_factor DEPRECATED!")
		depth = 3 * nBlocks + 4
		if (depth - 4) % 3:  raise Exception("Depth must be 3N + 4! (4,7,10,...) ")  # # layers in each denseblock
		N = (depth - 4) // 3
		print("N=%d" % N)
		do_dropout = True  # None  nil to disable dropout, non - zero number to enable dropout and set drop rate
		# dropRate = self.keep_prob # nil to disable dropout, non - zero number to enable dropout and set drop rate
		# channels before entering the first denseblock ??
		# set it to be comparable with growth rate ??

		growthRate = 12
		self.conv([3, 3, 1, nChannels])  # why this
		# self.conv([1, 3, 3, nChannels]) # and not this?

		# self.add(tf.nn.SpatialConvolution(3, nChannels, 3, 3, 1, 1, 1, 1))

		for i in range(N):
			self.addLayer(nChannels, growthRate, do_dropout)
			nChannels += growthRate
		self.addTransition(nChannels, nChannels, do_dropout)

		for i in range(N):
			self.addLayer(nChannels, growthRate, do_dropout)
			nChannels += growthRate
		self.addTransition(nChannels, nChannels, do_dropout)

		for i in range(N):
			self.addLayer(nChannels, growthRate, do_dropout)
			nChannels += growthRate

		self.batchnorm()
		self.add(tf.nn.relu(self.last_layer))
		# self.add(tf.nn.max_pool(self.last_layer, ksize=[1, 8, 8, 1], strides=[1, 2, 2, 1], padding='SAME'))
		# self.add(tf.nn.max_pool(self.last_layer, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='SAME'))
		# self.add(tf.nn.max_pool(self.last_layer, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME'))
		self.add(tf.nn.max_pool(self.last_layer, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME'))
		# self.add(tf.nn.SpatialAveragePooling(8, 8)).add(nn.Reshape(nChannels))

		shape = self.last_layer.get_shape()
		nBytes = shape[1] * shape[2] * shape[3]
		self.reshape([-1, int(nBytes)])  # ready for classification

	# Today's most performant vision models don't use fully connected layers anymore (they use convolutional blocks till the end and then some parameterless global averaging layer).
	# Fully connected layer
	def dense(self, hidden=1024, depth=1, activation=tf.nn.tanh, dropout=False, parent=-1, bn=False):  #
		if parent == -1: parent = self.last_layer
		if bn:
			print("dropout = False while using batchnorm")
			dropout = False
		shape = self.last_layer.get_shape()
		if shape and len(shape) > 2:
			if len(shape) == 3:
				self.last_width = int(shape[1] * shape[2])
			else:
				self.last_width = int(shape[1] * shape[2] * shape[3])
			if self.last_width == 0:
				raise Exception("self.last_width Must not be zero")
			print("reshaping ", shape, "to", self.last_width)
			parent = tf.reshape(parent, [-1, self.last_width])

		width = hidden
		while depth > 0:
			with tf.name_scope('Dense_{:d}'.format(hidden)) as scope:
				print("Dense ", self.last_width, width)
				nr = len(self.layers)
				U = tf.random_uniform([self.last_width, width], minval=-1. / width, maxval=1. / width)
				# U = np.random.rand(self.last_width, width) / (self.last_width + width)
				if self.last_width == width:
					U = closest_unitary(U / weight_divider)
				weights = tf.Variable(U, name="weights_dense_" + str(nr), dtype=tf.float32)
				bias = tf.Variable(tf.random_uniform([width], minval=-1. / width, maxval=1. / width), name="bias_dense")
				dense1 = tf.matmul(parent, weights, name='dense_' + str(nr)) + bias
				tf.summary.histogram('dense_' + str(nr), dense1)
				tf.summary.histogram('weights_' + str(nr), weights)
				tf.summary.histogram('bias_' + str(nr), bias)
				tf.summary.histogram('dense_' + str(nr) + '/sparsity', tf.nn.zero_fraction(dense1))
				tf.summary.histogram('weights_' + str(nr) + '/sparsity', tf.nn.zero_fraction(weights))
				if bn: dense1 = self.batchnorm(dense1, center=True)
				if activation: dense1 = activation(dense1)
				if dropout: dense1 = tf.nn.dropout(dense1, self.keep_prob)
				self.layers.append(dense1)
				self.last_layer = parent = dense1
				self.last_width = width
				depth = depth - 1
				self.last_shape = [-1, width]  # dense

	def conv2(self, shape, act=tf.nn.relu, pool=True, dropout=False, norm=True, name=None):
		with tf.name_scope('conv'):
			print("input  shape ", self.last_shape)
			print("conv   shape ", shape)
			# padding='VALID'
			conv = slim.conv2d(self.last_layer, shape[-1], [shape[1], shape[2]], 3, padding='SAME', scope=name)
			# if pool: conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			# if(pool): conv = slim.max_pool2d(conv, [2, 2], 1, scope='pool1')
			# if(pool): conv = slim.max_pool2d(conv, [3, 3], 2, scope='pool1')
			self.add(conv)

	# Convolution Layer
	def conv(self, shape, act=tf.nn.relu, pool=True, dropout=False, norm=True,
	         name=None):  # True why dropout bad in tensorflow??
		with tf.name_scope('conv'):
			print("input  shape ", self.last_shape)
			print("conv   shape ", shape)
			width = shape[-1]
			filters = tf.Variable(tf.random_normal(shape), name="filters")
			# filters = tf.Variable(tf.random_uniform(shape, minval=-1. / width, maxval=1. / width), name="filters")
			_bias = tf.Variable(tf.random_normal([shape[-1]]), name="bias")

			# # conv1 = conv2d('conv', _X, _weights, _bias)
			conv1 = tf.nn.bias_add(tf.nn.conv2d(self.last_layer, filter=filters, strides=[1, 1, 1, 1], padding='SAME'), _bias)
			if debug: tf.summary.histogram('conv_' + str(len(self.layers)), conv1)
			if act: conv1 = act(conv1)
			if pool: conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			if norm: conv1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
			if debug: tf.summary.histogram('norm_' + str(len(self.layers)), conv1)
			if dropout: conv1 = tf.nn.dropout(conv1, self.keep_prob)
			print("output shape ", conv1.get_shape())
			self.add(conv1)

	def rnn(self, num_hidden=42):
		# tf.contrib.rnn.BasicLSTMCell() OLD
		# tensorflow.python.ops.rnn_cell.BasicLSTMCell()
		# tensorflow.models.rnn.BasicLSTMCell()
		cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
		val, _ = tf.nn.dynamic_rnn(cell, self.last_layer, dtype=tf.float32)
		# Dropout does actually work quite well between recurrent units if you tie the dropout masks across time
		# val = tf.nn.dropout(val,self.keep_prob) # deprecated by batchnorm
		val = tf.transpose(val, [1, 0, 2])
		self.last = tf.gather(val, int(val.get_shape()[0]) - 1)

	def classifier(self, classes=0):  # Define loss and optimizer
		if not classes: classes = self.num_classes
		if not classes: raise Exception("Please specify num_classes")
		with tf.name_scope('prediction'):  # prediction
			if self.last_width != classes:
				# print("Automatically adding dense prediction")
				self.dense(hidden=classes, activation=None, dropout=False)
			# cross_entropy = -tf.reduce_sum(y_*y)
		with tf.name_scope('classifier'):
			y_ = self.target
			manual = False  # True
			if classes > 100:
				print("using sampled_softmax_loss")
				y = prediction = self.last_layer
				self.cost = tf.reduce_mean(tf.nn.sampled_softmax_loss(y, y_))  # for big vocab
			elif manual:
				# prediction = y =self.last_layer=tf.nn.softmax(self.last_layer)
				# self.cost = cross_entropy = -tf.reduce_sum(y_ * tf.log(y+ 1e-10)) # against NaN!
				prediction = y = tf.nn.log_softmax(self.last_layer)
				self.cost = cross_entropy = -tf.reduce_sum(y_ * y)
			else:
				self.output = y = prediction = self.last_layer
				self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))  # prediction, target
			tf.add_to_collection('outputs', self.output)

			# if not gpu:
			with tf.device(_cpu):
				tf.summary.scalar('cost', self.cost)
			# self.cost = tf.Print(self.cost , [self.cost ], "debug cost : ")
			# learning_scheme=self.learning_rate
			learning_scheme = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_size,
			                                             staircase=True)
			with tf.device(_cpu):
				tf.summary.scalar('learning_rate', learning_scheme)
			self.optimize = tf.train.AdamOptimizer(learning_scheme).minimize(self.cost)
			# self.optimizer = NeuralOptimizer(data=None, learning_rate=0.01, shared_loss=self.cost).minimize(self.cost) No good

			# Evaluate model
			correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.target, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			# if not gpu:
			tf.summary.scalar('accuracy', self.accuracy)
		# Launch the graph

	# noinspection PyAttributeOutsideInit
	def regression(self, dimensions, tolerance=3.):
		# self.dense(100)
		with tf.name_scope("regression"):
			self.dense(dimensions)
			self.y = tf.placeholder(tf.float32, [None, dimensions], name="target_y")  # self.batch_size
			print("REGRESSION 'accuracy' might not be indicative, watch loss")
			with tf.name_scope("train"):
				# self.learning_rate = tf.Variable(0.5, trainable=False)
				self.cost = tf.reduce_mean(tf.pow(self.y - self.last_layer, 2))
				self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
				self.accuracy = tf.maximum(0., 100 - tf.sqrt(self.cost)/tolerance)
				# self.accuracy = 1 - abs(self.y - self.last_layer)
				tf.add_to_collection('train_ops', [self.learning_rate, self.cost, self.optimize, self.accuracy])

	def debug_print(self, throughput, to_print=[]):
		return tf.cond(self.train_phase, lambda: throughput, lambda: tf.Print(throughput, to_print + [nop()], "OK!"))

	def next_batch(self, batch_size, session, test=False):
		# self.data either a generator or a data struct with properties .train/test.images/labels
		try:
			if test:
				test_images = self.data.test.images[:batch_size]
				test_labels = self.data.test.labels[:batch_size]
				return test_images, test_labels
			return self.data.train.next_batch(batch_size)
		except:
			try:
				return next(self.data)
			except:
				return next(self.data.train)

	def train(self, data=0, steps=-1, dropout=None, display_step=10, test_step=100, batch_size=10,
	          resume=save_step):  # epochs=-1,
		print("learning_rate: %f" % self.learning_rate)
		if data: self.data = data
		steps = 9999999 if steps < 0 else steps
		session = self.session
		# with tf.device(_cpu):
		# t = tf.verify_tensor_all_finite(t, msg)
		tf.add_check_numerics_ops()
		self.summaries = tf.summary.merge_all()
		self.summary_writer = tf.summary.FileWriter(current_logdir(), session.graph)  
		if not dropout: dropout = 1.  # keep all
		x = self.x
		y = self.y
		keep_prob = self.keep_prob
		if not resume or not self.resume(session):
			session.run([tf.global_variables_initializer()])
		saver = tf.train.Saver(tf.global_variables())
		snapshot = self.name + str(get_last_tensorboard_run_nr())
		step = 0  # show first
		while step < steps:
			batch_xs, batch_ys = self.next_batch(batch_size, session)
			# batch_xs=np.array(batch_xs).reshape([-1]+self.input_shape)
			# print("step %d \r" % step)# end=' ')
			# tf.train.shuffle_batch_join(example_list, batch_size, capacity=min_queue_size + batch_size * 16, min_queue_size)
			# Fit training using batch data
			feed_dict = {x: batch_xs, y: batch_ys, keep_prob: dropout, self.train_phase: True}
			loss, _ = session.run([self.cost, self.optimize], feed_dict=feed_dict)
			if step % display_step == 0:
				seconds = int(time.time()) - start
				# Calculate batch accuracy, loss
				feed = {x: batch_xs, y: batch_ys, keep_prob: 1., self.train_phase: False}
				acc, summary = session.run([self.accuracy, self.summaries], feed_dict=feed)
				# self.summary_writer.add_summary(summary, step) # only test summaries for smoother curve
				print("\rStep {:d} Loss= {:.6f} Accuracy= {:.3f} Time= {:d}s".format(step, loss, acc, seconds), end=' ')
				if str(loss) == "nan": return print("\nLoss gradiant explosion, exiting!")  # restore!
			if step % test_step == 0: self.test(step)
			if step % save_step == 0 and step > 0:
				print("SAVING snapshot %s" % snapshot)
				saver.save(session, checkpoint_dir + "/" + snapshot + ".ckpt", self.global_step)

			step += 1
		print("\nOptimization Finished!")
		self.test(step, number=10000)  # final test

	def test(self, step, number=400):  # 256 self.batch_size
		session = sess = self.session
		config = projector.ProjectorConfig()
		if visualize_cluster:  # EMBEDDINGs ++ https://github.com/tensorflow/tensorflow/issues/6322
			embedding = config.embeddings.add()  # You can add multiple embeddings. Here just one.

		run_metadata = tf.RunMetadata()
		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		# Calculate accuracy for 256 mnist test images

		test_images, test_labels = self.next_batch(number, session, test=True)
		# test_images = np.array(test_images).reshape([-1] + self.input_shape)

		feed_dict = {self.x: test_images, self.y: test_labels, self.keep_prob: 1., self.train_phase: False}
		# accuracy,summary= self.session.run([self.accuracy, self.summaries], feed_dict=feed_dict)
		accuracy, summary = session.run([self.accuracy, self.summaries], feed_dict, run_options, run_metadata)
		print('\t' * 3 + "Test Accuracy: {:.2f}".format( accuracy))
		self.summary_writer.add_run_metadata(run_metadata, 'step #%03d' % step)
		self.summary_writer.add_summary(summary, global_step=step)
		if accuracy == 1.0:
			print("OVERFIT OK. Early stopping")
			exit(0)

	def resume(self, session):
		checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
		if checkpoint:
			if self.name and not self.name in checkpoint:
				print("IGNORING checkpoint of other run : " + checkpoint + " !")
				checkpoint = None
		else:
			print("NO checkpoint, nothing to resume")
		if checkpoint:
			print("LOADING " + checkpoint + " !")
			try:
				persister = tf.train.Saver(tf.global_variables())
				persister.restore(session, checkpoint)
				print("resume checkpoint successful!")
				return True
			except Exception as ex:
				print(ex)
				print("CANNOT LOAD checkpoint %s !" % checkpoint)
		return False

	def restore(self):  # name
		# if not session: session= tf.Session()
		self.session = tf.Session()
		checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			print("Restoring old model from meta graph")
			loader = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + ".meta")
		else:
			print("No model from meta graph, nothing to restore")
			return self
		self.session.run(tf.global_variables_initializer())
		print("loading checkpoint %s" % checkpoint.model_checkpoint_path)
		try:
			loader.restore(self.session, tf.train.latest_checkpoint(checkpoint_dir))
		except:
			pass
		# loader.restore(self.session , checkpoint) #Unable to get element from the feed as bytes!  HUH??
		self.input = self.x = tf.get_collection('inputs')[0]
		self.target = self.y = tf.get_collection('targets')[0]
		self.output = self.last_layer = tf.get_collection('outputs')[0]
		self.dropout_keep_prob = self.session.graph.get_tensor_by_name("state/dropout_keep_prob:0")  # :0 WTF!?!?!
		self.train_phase = self.session.graph.get_tensor_by_name(name='state/train_phase:0')
		return self

	def predict(self, eval_data=None, model=None):
		if eval_data is None:
			eval_data = np.random.random(self.input_shape)
		feed_dict = {self.x: [eval_data], self.dropout_keep_prob: 1.0, self.train_phase: False}
		result = self.session.run([self.output], feed_dict)
		best = np.argmax(result)
		# print("prediction: %s" % result)
		print("predicted: %s" % best)
		return best
