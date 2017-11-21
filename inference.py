import tensorflow as tf
import network
import os
import cv2
import glob
import numpy as np
import math

data_dir = '.\\'    # TODO: determine data_dir path
label_dir = data_dir + 'label\\'
ldata = glob.glob(label_dir + '\*.npy')
check_dir = data_dir + 'check\\'
c_p = 0.2


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

    # TODO : Confirm type of float
    # Create placeholders for import
    images = tf.placeholder(tf.float32, (1, image_size, image_size))
    labels = tf.placeholder(tf.float32, (1, S, S, last_output))
    obj = tf.placeholder(tf.float32, (1, S, S))
    noobj = tf.placeholder(tf.float32, (1, S, S))
    keep_prob = tf.placeholder(tf.float32)

    # Build a graph that computes the logits predictions from the inference model.
    logits, weight_sum = network.inference(images, keep_prob)

    # Calculate loss
    # loss = network.loss(logits, labels, obj, noobj)
    l1, l2, l3, l4 = network.loss(logits, labels, obj, noobj, 1)

    """
    up to here, building a model, not starting training.
    """

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        cwd = os.getcwd()
        if os.path.isfile(cwd+"\\model.ckpt.index"):
            print("restore!!")
            saver.restore(sess, cwd+"\\model.ckpt")
        for label_path in ldata[:30]:
            im = np.zeros((1, image_size, image_size))
            label = np.zeros((1, S, S, last_output))
            ob = np.zeros((1, S, S))
            noob = np.zeros((1, S, S))

            im[0], label[0] = np.load(label_path)
            ob[0] = label[:, :, :, 4]
            noob[0] = - ob + 1.0

            res = sess.run(logits, feed_dict={images: im, labels: label, obj: ob, noobj: noob, keep_prob: 1.0})
            ll1, ll2, ll3, ll4 = sess.run((l1, l2, l3, l4), {images: im, labels: label, obj: ob, noobj: noob,
                                                                keep_prob: 1.0})

            print(label_path)
            print('loss: %f, %f, %f, %f' % (ll1, ll2, ll3, ll4))

            image = im[0].astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            comp = image_size / S
            for i in range(S):
                for j in range(S):
                    x = res[0][i][j][0] * comp + comp * i
                    y = res[0][i][j][1] * comp + comp * j
                    w = res[0][i][j][2] * image_size
                    h = res[0][i][j][3] * image_size
                    c = res[0][i][j][4]

                    if c > c_p:
                        p1 = (math.floor(x - h / 2), math.floor(y - w / 2))
                        p2 = (math.floor(x - h / 2), math.ceil(y + w / 2))
                        p3 = (math.ceil(x + h / 2), math.floor(y - w / 2))
                        p4 = (math.ceil(x + h / 2), math.ceil(y + w / 2))
                        cv2.line(image, p1, p2, (0, 255, 0), 2)
                        cv2.line(image, p1, p3, (0, 255, 0), 2)
                        cv2.line(image, p2, p4, (0, 255, 0), 2)
                        cv2.line(image, p3, p4, (0, 255, 0), 2)

            cv2.imwrite(check_dir + '\\' + os.path.splitext(os.path.basename(label_path))[0] + '.jpg', image)


if __name__ == '__main__':
    tf.app.run(main=main)
