import glob
import numpy as np
import network
import os
import random

data_dir = '.\\'    # TODO: determine data_dir path
label_dir = data_dir + 'label\\'

ldata = glob.glob(label_dir + '\*.npy')
n_data = len(ldata)   # the number of data

image_size = network.IMAGE_SIZE
S = network.S
B = network.B
last_output = network.last_output

count = 0


def next_batch(batch_size, index):
    """
    :param batch_size: random batch_size for one iteration.
    :return: tuple of images, labels, objs presentating whether an obj exists or not, no_objs presentating not whether.
    """

    index = random.sample(range(n_data), batch_size)
    images = np.zeros((batch_size, image_size, image_size))
    labels = np.zeros((batch_size, S, S, last_output))
    objs = np.zeros((batch_size, S, S))
    no_objs = np.zeros((batch_size, S, S))
    for i in range(batch_size):
        label_path = label_dir + os.path.basename(ldata[index[i]])
        image, label = np.load(label_path)   # The image and label are set.
        image.astype(np.float32)

        # TODO: Confirm type of image(numpy array) (numpy to tensor should be bloadcasting...)
        images[i] = image
        labels[i] = label

        obj = label[:, :, 4]
        no_obj = - obj + 1.0
        objs[i] = obj
        no_objs[i] = no_obj

    return images, labels, objs, no_objs


# Only in case this programs is called directly, the script below is executed.
"""
if __name__ == '__main__':

    ldata = glob.glob(label_dir + '\*.npy')
    n_data = len(ldata)   # the number of data
"""

