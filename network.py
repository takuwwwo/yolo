"""
Builds the detecting rectangles for flowcharts by YOLO algorithm.
"""

import tensorflow as tf
import tools

# The rectangle class has 1 classes.
C = 1

# The last output image size
S = 24

# The number of obejects which can be detected in a cell
B = 1

# Last output size
last_output = B*5  # + C

# The input images should be resized to 448*448
IMAGE_SIZE = S * 16

FLAGS = None


def yolorelu(x):
    return tf.add(tf.nn.relu(tf.scalar_mul(0.9, x)), tf.scalar_mul(0.1, x))


def likesqrt(x):
    return tf.sign(x) * tf.sqrt(tf.abs(x))


def inference(images, keep_prob):
    """
    Builds the rectangle detecting model

    Args:
        images: Images placeholder, from inputs()
        keep_prob: The rate of dropout.

    Returns:
        ReLU_like : Output inference network
    """
    x_image = tf.reshape(images, (-1, IMAGE_SIZE, IMAGE_SIZE, 1))
    conv1channel = 16   # 16
    conv2channel = 16   # 32
    conv3channel = 32   # 64
    conv4channel = 64   # 128
    conv5channel = 128  # 256
    conv6channel = 256  # 512
    conv7channel = 512  # 1024
    conv8channel = 512  # 1024
    conv9channel = 128  # 1024

    fc1_length = 1024
    fc2_length = 4096

    weight_sum = 0.0

    tf.summary.image('image', x_image, tools.batch_size)
    with tf.variable_scope('conv1'):
        weights = tools.weight_variables([3, 3, 1, conv1channel], name="W")
        biases = tools.bias_variables([conv1channel], name="B")
        conv1 = yolorelu(tools.conv2d(x_image, weights) + biases)
        pool1 = tools.max_pool_2x2(conv1)
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activation", conv1)

    with tf.variable_scope('conv2'):
        weights = tools.weight_variables([3, 3, conv1channel, conv2channel], name="W")
        biases = tools.bias_variables([conv2channel], name="B")
        conv2 = yolorelu(tools.conv2d(pool1, weights) + biases)
        pool2 = tools.max_pool_2x2(conv2)
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activation", conv2)

    with tf.variable_scope('conv3'):
        weights = tools.weight_variables([3, 3, conv2channel, conv3channel], name="W")
        biases = tools.bias_variables([conv3channel], name="B")
        conv3 = yolorelu(tools.conv2d(pool2, weights) + biases)
        pool3 = tools.max_pool_2x2(conv3)
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activation", conv3)

    with tf.variable_scope('conv4'):
        weights = tools.weight_variables([3, 3, conv3channel, conv4channel], name="W")
        biases = tools.bias_variables([conv4channel], name="B")
        conv4 = yolorelu(tools.conv2d(pool3, weights) + biases)
        pool4 = tools.max_pool_2x2(conv4)
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activation", conv4)

    with tf.variable_scope('conv5'):
        weights = tools.weight_variables([3, 3, conv4channel, conv5channel], name="W")
        biases = tools.bias_variables([conv5channel], name="B")
        conv5 = yolorelu(tools.conv2d(pool4, weights) + biases)
        # pool5 = tools.max_pool_2x2(conv5)
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activation", conv5)

    with tf.variable_scope('conv6'):
        weights = tools.weight_variables([3, 3, conv5channel, conv6channel], name="W")
        biases = tools.bias_variables([conv6channel], name="B")
        conv6 = yolorelu(tools.conv2d(conv5, weights) + biases)
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activation", conv6)

    with tf.variable_scope('conv7'):
        weights = tools.weight_variables([3, 3, conv6channel, conv7channel], name="W")
        biases = tools.bias_variables([conv7channel], name="B")
        conv7 = yolorelu(tools.conv2d(conv6, weights) + biases)
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activation", conv7)

    with tf.variable_scope('conv8'):
        weights = tools.weight_variables([3, 3, conv7channel, conv8channel], name="W")
        biases = tools.bias_variables([conv8channel], name="B")
        conv8 = yolorelu(tools.conv2d(conv7, weights) + biases)
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activation", conv8)

    with tf.variable_scope('conv9'):
        weights = tools.weight_variables([3, 3, conv8channel, conv9channel], name="W")
        biases = tools.bias_variables([conv9channel], name="B")
        conv9 = yolorelu(tools.conv2d(conv8, weights) + biases)
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activation", conv9)

    """
    with tf.variable_scope('conv10'):
        weights = tools.weight_variables([3, 3, conv9channel, last_output], name="W")
        biases = tools.bias_variables([last_output], name="B")
        y = tools.conv2d(conv9, weights) + biases
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activation", y)
    """

    with tf.variable_scope('fc1'):
        weights = tools.weight_variables([S*S*conv9channel, fc1_length], name="W")
        biases = tools.bias_variables([fc1_length], name="B")
        flat = tf.reshape(conv9, [-1, S*S*conv9channel])
        fc1 = tf.matmul(flat, weights) + biases
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activation", fc1)

    with tf.variable_scope('fc2'):
        weights = tools.weight_variables([fc1_length, fc2_length], name="W")
        biases = tools.bias_variables([fc2_length], name="B")
        fc2 = yolorelu(tf.matmul(fc1, weights) + biases)
        fc2_drop = tf.nn.dropout(fc2, keep_prob)
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activation", fc1)

    with tf.variable_scope('fc3'):
        weights = tools.weight_variables([fc2_length, S*S*last_output], name="W")
        biases = tools.bias_variables([S*S*last_output], name="B")
        y = tf.reshape(tf.matmul(fc2_drop, weights) + biases, [-1, S, S, last_output])
        weight_sum += tf.nn.l2_loss(weights)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("x", y[:, :, :, 0])
        tf.summary.histogram("y", y[:, :, :, 1])
        tf.summary.histogram("w", y[:, :, :, 2])
        tf.summary.histogram("h", y[:, :, :, 3])
        tf.summary.histogram("confidence", y[:, :, :, 4])

    return y, weight_sum


def loss(logits, labels, obj, noobj, batch_size, l_obj=1.0, l_noobj=0.1):   # TODO: confirm
    """
    args:
        logits: Logits tensor, float - [x, y, w, h. C, x2, y2, w2, h2, c2]
        labels: Logits tensor, float - [x, y, w, h. C, x2, y2, w2, h2, c2]
        obj: matrix showing whether an object exists in the cell or not - [2, S, S]
        noobj: matrix showing whether an object doesn't exist in the cell or not - [2, S, S]
        batch_size: batch size
        l_obj, l_noobj: hyper parameter
    """

    x, y, w, h, c = 0, 1, 2, 3, 4
    # p1 = 5
    with tf.variable_scope('loss1'):
        l1 = l_obj * tf.reduce_sum(obj *
                                    (tf.square((logits[:, :, :, x] - labels[:, :, :, x])) +
                                     tf.square((logits[:, :, :, y] - labels[:, :, :, y]))))

    with tf.variable_scope('loss2'):
        l2 = l_obj * tf.reduce_sum(obj *
                                    (tf.square((likesqrt(logits[:, :, :, w]) - tf.sqrt(labels[:, :, :, w]))) +
                                     tf.square((likesqrt(logits[:, :, :, h]) - tf.sqrt(labels[:, :, :, h])))))

    with tf.variable_scope('loss3'):
        l3 = tf.reduce_sum(obj * tf.square(logits[:, :, :, c] - labels[:, :, :, c]))

    with tf.variable_scope('loss4'):
        l4 = l_noobj * tf.reduce_sum(noobj *
                                      (tf.square(logits[:, :, :, c] - labels[:, :, :, c])))

    l = (l1 + l2 + l3 + l4) / batch_size
    """

    l5 = tf.reduce_sum(obj *
                       tf.reduce_sum(tf.square(logits[:, :, :, p1] - labels[:, :, :, p1])))
    """

    return l


def training(loss2, w, initial_learning_rate):
    """
    Args:
        loss2: Loss value
        initial_learning_rate: learning rate
    """

    global_step = tf.Variable(0, trainable=False)
    boundaries = [501, 2501, 10001, 20001, 30001, 40001, 50001, 70001, 90001, 120001, 150001, 200001]
    values = [initial_learning_rate * 0.5, initial_learning_rate,
              initial_learning_rate * 1.5, initial_learning_rate * 2, initial_learning_rate * 3,
              initial_learning_rate * 2, initial_learning_rate * 1.5,
              initial_learning_rate, initial_learning_rate * 0.5, initial_learning_rate * 0.3,
              initial_learning_rate * 0.1, initial_learning_rate * 0.04, initial_learning_rate * 0.01]
    """
    boundaries = [101, 1001, 2001, 3001, 4001, 6001, 8001, 10001, 12001, 14001, 18001, 24001, 30001, 40001]
    values = [initial_learning_rate*0.5, initial_learning_rate, initial_learning_rate*1.5,
              initial_learning_rate*2, initial_learning_rate*2.5, initial_learning_rate*3,
              initial_learning_rate*2.5, initial_learning_rate*2, initial_learning_rate*1.5,
              initial_learning_rate, initial_learning_rate*0.5, initial_learning_rate*0.3,
              initial_learning_rate*0.1, initial_learning_rate*0.04, initial_learning_rate*0.01]
    """
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('l', loss2)
    tf.summary.scalar('weight', w)
    tf.summary.scalar('loss', loss2+w)

    # Create the gradient descent optimizer with the given learning rate
    opt = tf.train.AdamOptimizer(learning_rate)
    grads = opt.compute_gradients(loss2+w)

    # Apply gradients
    optimizer = opt.apply_gradients(grads, global_step=global_step)
    """
    optimizer.minimize is function combining compute_gradients and apply_gradients.
    """

    global_step = global_step + 1
    tf.summary.scalar('step', global_step)

    return optimizer

