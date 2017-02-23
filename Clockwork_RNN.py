import tensorflow as tf
# import tensorlayer as tl
class ClockworkLayer(tl.layers.RNNLayer):
	""" A clockwork RNN layer.

	As done in the original paper, we restrict ourselves to an exponential
	series of periods. As noted in the paper, this lets W_H and W_I be
	contiguous, and the implementation is therefore much simpler.

	This is based on Jan Koutnik et al.: A Clockwork RNN.
	arXiv preprint arXiv:1402.3511. 2014.

	See `RNNLayer`.

	**kwargs:
		num_periods: An integer. The periods will be `2 ** np.arange(num_periods)`.
			8 by default.
	"""

	def __init__(self, *args, **kwargs):
		kwargs.setdefault('num_periods', 8)
		super().__init__(*args, **kwargs)

	def _compute_states(self):
		units_per_period, remainder = divmod(self.num_hidden_units, self.num_periods)
		if remainder != 0:
			raise ValueError('Current implementation requires num_hidden_units to be divisible by num_periods.')

		_inputs = tf.transpose(self.inputs, [1, 0, 2])
		x_ta = tf.TensorArray(tf.float32, size=self.length).unstack(_inputs)
		h_ta = tf.TensorArray(tf.float32, size=self.length)

		def cond(t, h, h_ta):
			return tf.less(t, self.length)

		def body(t, h, h_ta):
			x = x_ta.read(t)
			num_units, input_size = self.num_hidden_units, self.input_size
			periods = tf.constant(2**np.arange(self.num_periods), dtype=tf.int32)

			# We need to transpose everything in Figure 2 of the paper.
			weight_mask = utils.block_triu_mask(units_per_period, self.num_periods).T
			weight_mask = tf.constant(weight_mask, dtype=tf.float32)

			active_period_mask = tf.to_int32(tf.equal(tf.mod(t, periods), 0))
			num_active_periods = tf.reduce_sum(active_period_mask)
			num_active_units = num_active_periods * units_per_period

			W_h = tf.get_variable('W_h', shape=[num_units, num_units], initializer=self.non_square_initializer)
			W_x = tf.get_variable('W_x', shape=[input_size, num_units], initializer=self.non_square_initializer)
			b = tf.get_variable('b', shape=[num_units], initializer=self.bias_initializer)

			# W_h was created fully for simplicity and efficiency, but only its
			# lower-block-triangular version stores clockwork parameters.
			W_h = weight_mask * W_h

			# Shutting off parts of h is handled using the W_h mask above. Therefore
			# we will always multiply by all of h (and therefore use the entire first
			# axis of W_h). None of x is ever shut off, so we do the same. Finally,
			# we only want outputs for active states, which correspond to the left
			# sides of W_h, W_x, and b.
			W_h = W_h[:, :num_active_units]
			W_x = W_x[:, :num_active_units]
			b = b[:num_active_units]

			h_new_active = self.activation(tf.matmul(h, W_h) + tf.matmul(x, W_x) + b)
			h_new_inactive = h[:, num_active_units:]
			h_new = tf.concat_v2([h_new_active, h_new_inactive], axis=1)
			h_new = tf.reshape(h_new, [self.batch_size, self.num_hidden_units])

			h_ta_new = h_ta.write(t, h_new)
			return t + 1, h_new, h_ta_new

		t = tf.constant(0)
		h = tf.squeeze(self.initial_states, [1])
		_, _, h_ta = tf.while_loop(cond, body, [t, h, h_ta])

		states = tf.transpose(h_ta.stack(), [1, 0, 2], name='states')
		outputs = tf.identity(states, name='outputs')
		return outputs, states
