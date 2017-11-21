import tensorflow as tf
import network
import yolo_input
import tools
import os
import math

learning_rate = 0.0003

# weight_decay rate
weight_decay = 0.0005


def main(_):
    # The image size of input image
    image_size = network.IMAGE_SIZE

    # The number of class.
    C = network.C

    # The last output image size
    S = network.S

    # The number of obejects which can be detected in a cell
    B = network.B

    # The depth of last layer.
    last_output = network.last_output

    # batch size
    batch_size = tools.batch_size

    # TODO : Confirm type of float
    # Create placeholders for import
    images = tf.placeholder(tf.float32, (batch_size, image_size, image_size))
    labels = tf.placeholder(tf.float32, (batch_size, S, S, last_output))
    obj = tf.placeholder(tf.float32, (batch_size, S, S))
    noobj = tf.placeholder(tf.float32, (batch_size, S, S))
    keep_prob = tf.placeholder(tf.float32)

    # Build a graph that computes the logits predictions from the inference model.
    logits, weight_sum = network.inference(images, keep_prob)
    w = weight_sum * weight_decay

    # Calculate loss
    # loss = network.loss(logits, labels, obj, noobj)
    l = network.loss(logits, labels, obj, noobj, batch_size)

    # Build a graph that trains the model with one batch of examples and updates the model parameters
    train_op = network.training(l, w, learning_rate)

    """
    up to here, building a model, not starting training.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log", graph=sess.graph)

        saver = tf.train.Saver()
        cwd = os.getcwd()
        if os.path.isfile(cwd+"\\model.ckpt.index"):
            print("restore!!")
            saver.restore(sess, cwd+"\\model.ckpt")

        for i in range(250000):                       # TODO
            im, l, ob, noob = yolo_input.next_batch(batch_size, i)  # TODO
            if i % 10 == 0:
                s = sess.run(merged_summary, feed_dict={images: im, labels: l, obj: ob, noobj: noob,
                                              keep_prob: 1.0})
                writer.add_summary(s, i)
            _ = sess.run(train_op, feed_dict={images: im, labels: l, obj: ob, noobj: noob, keep_prob: 0.5})
            if i % 100 == 99:
                print('step: %d: saved!' % i)

        saver.save(sess, cwd + "\\model.ckpt")

if __name__ == '__main__':
    tf.app.run(main=main)

