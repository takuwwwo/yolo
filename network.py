"""
Builds the detecting rectangles for flowcharts by YOLO algorithm.

"""

import argparse
import sys
import tensorflow as tf
import tools

# The rectangle class has 2 classes.
C = 2

# The last output image size
S = 7

# The number of obejects which can be detected in a cell
B = 1

# Last output size
last_output = B*5 + C

# The input images should be resized to 448*448
IMAGE_SIZE = 448

FLAGS = None

def yolorelu(x):
    return tf.add(tf.nn.relu(tf.scalar_multiply(0.9, x)), tf.scalar_multiply(0.1, x))

def deepnn(x):
    # The number of channels in each layers
    conv1channel = 64
    conv2channel = 64
    conv3channel = 128
    conv4channel = 256
    conv5channel = 256
    conv6channel = 256
    fc1_length = 1024
    last_output = B * 5 + C

    #1st layer
    W_conv1 = tools.weight_variables([7, 7, 1, conv1channel])
    b_conv1 = tools.bias_variable([conv1channel])
    h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)

    h_pool1 = tools.max_pool_2x2(h_conv1)

    #2nd layer
    W_conv2 = tools.weight_variables([3, 3, conv1channel, conv2channel])
    b_conv2 = tools.bias_variable([conv2channel])
    h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)

    h_pool2 = tools.max_pool_2x2(h_conv2)

    #3rd layer
    W_conv3 = tools.weight_variables([3, 3, conv2channel, conv3channel])
    b_conv3 = tools.bias_variable([conv3channel])
    h_conv3 = tf.nn.relu(tools.conv2d(h_pool2, W_conv3) + b_conv3)

    h_pool3 = tools.max_pool_2x2(h_conv3)

    #4th layer
    W_conv4 = tools.weight_variables([3, 3, conv3channel, conv4channel])
    b_conv4 = tools.bias_variable([conv4channel])
    h_conv4 = tf.nn.relu(tools.conv2d(h_pool3, W_conv4) + b_conv4)

    h_pool4 = tools.max_pool_2x2(h_conv4)

    #5th layer
    W_conv5 = tools.weight_variables([3, 3, conv4channel, conv5channel])
    b_conv5 = tools.bias_variable([conv5channel])
    h_conv5 = tf.nn.relu(tools.conv2d(h_pool4, W_conv5) + b_conv5)

    h_pool5 = tools.max_pool_2x2(h_conv5)

    #6th layer
    W_conv6 = tools.weight_variables([3, 3, conv5channel, conv6channel])
    b_conv6 = tools.bias_variable([conv6channel])
    h_conv6 = tf.nn.relu(tools.conv2d(h_pool5, W_conv6) + b_conv6)

    h_pool6 = tools.max_pool_2x2(h_conv6)

    #7th layer (Fully connected layer1)
    W_fc1 = tools.weight_variables([7*7*conv6_channel, fc1_length])
    b_fc1 = tools.bias_variable([fc1_length])

    h_pool2_flat = tf.reshape([h_pool6, [-1, 7*7*conv6_channel]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adatation of features
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #8th layer (Fully connected layer2)
    W_fc2 = weight_variable([fc1_length, S*S*last_output])
    b_fc2 = bias_variable([S*S*last_output])

    y_conv = tf.reshape(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, [S, S, last_output])
    return y_conv, keep_prob


def inference(images):
    """
    Builds the rectangle detecting model

    Args:
        images: Images placeholder, from inputs()
        hidden(n)_units: Size of the nth hidden layer

    Returns:
        ReLU_like : Output inference network
    """
    with tf.variable_scope('conv1') as scope:
        weights = tools.weight_variables([7, 7, 1, conv1channel])
        biases = tools.bias_variable([conv1channel])
        conv1 = ReLU(tools.conv2d(x_image, weights) + biases)
    pool1 = tools.max_pool_2x2(conv1)

    with tf.variable_scope('conv2') as scope:
        weights = tools.weight_variables([3, 3, conv1channel, conv2channel])
        biases = tools.bias_variable([conv2channel])
        conv2 = ReLU(tools.conv2d(pool1, weights) + biases)
    pool2 = tools.max_pool_2x2(conv2)

    with tf.variable_scope('conv3') as scope:
        weights = tools.weight_variables([3, 3, conv2channel, conv3channel])
        biases = tools.bias_variable([conv3channel])
        conv3 = ReLU(tools.conv2d(pool2, weights) + biases)
    pool3 = tools.max_pool_2x2(conv3)

    with tf.variable_scope('conv4') as scope:
        weights = tools.weight_variables([3, 3, conv3channel, conv4channel])
        biases = tools.bias_variable([conv4channel])
        conv4 = ReLU(tools.conv2d(pool3, weights) + biases)
    pool4 = tools.max_pool_2x2(conv4)

    with tf.variable_scope('conv5') as scope:
        weights = tools.weight_variables([3, 3, conv4channel, conv5channel])
        biases = tools.bias_variable([conv5channel])
        conv5 = ReLU(tools.conv2d(pool4, weights) + biases)
    pool5 = tools.max_pool_2x2(conv5)

    with tf.variable_scope('conv6') as scope:
        weights = tools.weight_variables([3, 3, conv5channel, conv6channel])
        biases = tools.bias_variable([conv6channel])
        conv6 = ReLU(tools.conv2d(pool5, weights) + biases)
    pool6 = tools.max_pool_2x2(conv6)

    with tf.variable_scope('fc1') as scope:
        weights = tools.weight_variables([7*7*conv6_channel fc1_length])
        biases = tools.bias_variable([fc1_length])
        flat = tf.reshape(pool6, [-1, 7*7*conv6_channel])
        fc1 = ReLU(tf.matmul(flat, weights) + biases)


    keep_prob = tf.placeholder(tf.float32)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    with tf.variable_scope('fc2') as scope:
        weights = tools.weight_variables([fc1_length, S*S*last_output])
        biases = tools.bias_variable([S*S*last_output])
        y = tf.reshape(tf.matmul(fc1_drop, weights) + biases, [S, S, last_output])

    return y

def loss(logits, labels, obj, noobj, l_obj = 5.0, l_noobj = 0.5): #TODO: confirm
    """
    args:
        logits: Logits tensor, float - [x, y, w, h. C, p[2]]
        labels: Labels tensor, float - [x, y, w, h. C, p[2]]
        obj: matrix showing whether an object exists in the cell or not
        noobj: matrix showing whether an object doesn't exist in the cell or not
        l_obj, l_noobj: hyper parameter
    """

    x, y, w, h, C = 0, 1, 2, 3, 4; p = [5, 6]
    l1 = l_obj * tf.reduce_sum(obj *
                               (tf.square((logits[x] - labels[x])) +
                                tf.square((logits[y] - labels[y]))))

    l2 = l_obj * tf.reduce_sum(obj *
                               (tf.square((tf.sqrt(logits[w]) - tf.sqrt(labels[w]))) +
                                tf.square((tf.sqrt(logits[h]) - tf.sqrt(labels[h])))))

    l3 = tf.reduce_sum(obj *
                       tf.square(logits[C] - labels[C]))

    l4 = l_noobj * tf.reduce_sum(noobj *
                                 (tf.square(logits[C] - labels[C])))

    l5 = tf.reduce_sum(obj*
                       tf.reduce_sum(tf.square(logits[p] - labels[p]), axis = 0))

    return l1 + l2 + l3 + l4 + l5

def training(loss, learning_rate):
    """
    Args:
        loss: Loss value
        learning_rate: learning rate

    """

    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)

    # Create the gradient descent optimizer with the given learning rate
    opt = tf.train.AdamOptimizer(learning_rate)
    grads = opt.compute_gradients(loss)

    # Apply gradients
    optimizer = opt.apply_gradients(grads, global_step=global_step)

    return optimizer












print(inference(1,2,3,4,5,6,7,8,9))
