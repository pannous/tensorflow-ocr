from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell

np.random.seed(seed= 7)
n_iterations = 300


def generate_sequences(sequence_num, sequence_length, batch_size):
    x_data = np.random.uniform(0, 1, size=(sequence_num / batch_size, sequence_length, batch_size, 1))
    y_data = []
    for x in x_data:
        sequence = [x[0]]
        for index in xrange(1, len(x)):
            sequence.append(x[0] * x[index]) # m u l OK
            # sequence.append(x[0] + x[index]) # sum DOESNT
            # sequence.append(1-x[index-1])
            # sequence.append(1.-x[index])
            # sequence.append(x[index-1] * x[index]) # m u l OK
            # sequence.append(x[index] *np.average(x[index-1]))
            # sequence.append(x[0] *np.average(x[index]))
            # sequence.append(x[0] *np.max(x[index]))
            # sequence.append([1]*sequence_length *np.max(x[index]))
            # sequence.append([sequence.max(axis=0)])
            # max=np.max(sequence,axis=None, out=None, keepdims=True)#/2
            # sequence.append([np.maximum(sequence, max)])
            # sequence.append([np.max(sequence, axis=0)])
            # candidates_for_min = sequence[1:]
            # sequence.append([np.min(candidates_for_min, axis=0)])
        y_data.append(sequence)
    return x_data, y_data


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def main():
    sequence_num = 100 # datapoints_number
    sequence_length = 10
    batch_size = 10
    data_point_dim = 1
    if sequence_num % float(batch_size) != 0:
        raise ValueError('Number of samples must be divisible with batch size')
    inputs, outputs = generate_sequences(sequence_num, sequence_length, batch_size)

    input_dim = len(inputs[0][0])
    output_dim = len(outputs[0][0])

    encoder_inputs = [tf.placeholder(tf.float32, shape=[batch_size, data_point_dim]) for _ in xrange(input_dim)]
    decoder_inputs = [tf.placeholder(tf.float32, shape=[batch_size, data_point_dim]) for _ in xrange(output_dim)]
    lstm_cell = rnn_cell.BasicLSTMCell(data_point_dim, state_is_tuple=True)
    model_outputs, states = seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, lstm_cell)

    reshaped_outputs = tf.reshape(model_outputs, [-1])
    reshaped_results = tf.reshape(decoder_inputs, [-1])

    cost = tf.reduce_sum(tf.squared_difference(reshaped_outputs, reshaped_results))
    variable_summaries(cost, 'cost')
    step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    merged = tf.summary.merge_all()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("/tmp/tensor/train", session.graph)
        costs = []
        for i in xrange(n_iterations):
            batch_costs = []
            summary = None
            for batch_inputs, batch_outputs in zip(inputs, outputs):
                # x_list = {key: value for (key, value) in zip(encoder_inputs, batch_inputs)}
                # y_list = {key: value for (key, value) in zip(decoder_inputs, batch_outputs)}
                # feed_dict = dict(x_list.items() + y_list.items())
                feed_dict = {encoder_inputs: batch_inputs, decoder_inputs: batch_outputs}
                # summary, err, _ = session.run([merged, cost, step], )
                summary, err, _ = session.run([merged, cost, step], feed_dict)
                batch_costs.append(err)
                print("err",err)
            if summary is not None:
                writer.add_summary(summary, i)
            costs.append(np.average(batch_costs, axis=0))

    # predict = session.run([model_outputs], feed_dict=dict(x_list.items() + y_list.items()))
    # print("predict",predict)
    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
	main()
