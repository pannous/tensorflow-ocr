# tf.train.slice_input_producer(tensor_list, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, name=None)

#  tensor flow now supports generalized dilated atrous co'lusion !!
# -    x = tf.nn.conv3d(x, kernel, strides, padding)
# +    x = tf.nn.convolution(x, kernel, padding,strides=strides, dilation_rate=filter_dilation)
# https://github.com/fchollet/keras/pull/4115/files

# AtrousConv1D = AtrousConvolution1D
# YAY!!!!

# https://github.com/nrTQgc/deep-anpr/commit/adfbbe7f3deeaa39bdecc36d8398434b76fdf211
#  fork confusion : ^^  13 days ago BUUUUT: # https://github.com/nrTQgc/deep-anpr  two months ago!?!?!?!
