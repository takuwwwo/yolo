import tensorflow as tf

batch_size = 16


def weight_variables(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.0001)
    return tf.Variable(initial, name=name)


def bias_variables(shape, name):
    initial = tf.constant(0.00001, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')