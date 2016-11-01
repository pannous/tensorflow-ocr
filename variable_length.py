import tensorflow as tf
import numpy as np

if __name__ == '__main__':
  np.random.seed(1)
  size = 100
  batch_size= 100
  n_steps = 45
  seq_width = 50

  initializer = tf.random_uniform_initializer(-1,1)

  seq_input = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
    #sequence we will provide at runtime
  early_stop = tf.placeholder(tf.int32)
    #what timestep we want to stop at

  inputs = [tf.reshape(i, (batch_size, seq_width)) for i in tf.split(0, n_steps, seq_input)]
    #inputs for rnn needs to be a list, each item being a timestep.
    #we need to split our input into each timestep, and reshape it because split keeps dims by default

  cell = tf.nn.rnn_cell.LSTMCell(size, seq_width, initializer=initializer)
  initial_state = cell.zero_state(batch_size, tf.float32)
  outputs, states = tf.nn.rnn(cell, inputs, initial_state=initial_state, sequence_length=early_stop)
    #set up lstm

  iop = tf.initialize_all_variables()
    #create initialize op, this needs to be run by the session!
  session = tf.Session()
  session.run(iop)
    #actually initialize, if you don't do this you get errors about uninitialized stuff

  feed = {early_stop:100, seq_input:np.random.rand(n_steps, batch_size, seq_width).astype('float32')}
    #define our feeds.
    #early_stop can be varied, but seq_input needs to match the shape that was defined earlier

  outs = session.run(outputs, feed_dict=feed)
    #run once
    #output is a list, each item being a single timestep. Items at t>early_stop are all 0s
  print type(outs)
  print len(outs)
