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

# The input images should be resized to 448*448
IMAGE_SIZE = 448

FLAGS = None

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


def inference(images,
              hidden1_units, hidden2_units, hidden3_units, hidden4_units,
              hidden5_units, hidden6_units, hidden7_units, hidden8_units):
    """
    Builds the rectangle detecting model

    Args:
        images: Images placeholder, from inputs()
        hidden(n)_units: Size of the nth hidden layer

    Returns:
        ReLU_like : Output inference network
    """




    return 1

print(inference(1,2,3,4,5,6,7,8,9))
