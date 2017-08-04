import tensorflow as tf
import glob
import numpy as np

data_dir = ''
image_dir = data_dir + '\image'
label_dir = data_dir + '\label'
n_data = None

def next_batch(batch_size):
    index = np.random.random_integers(1, n_data, batch_size)



if __name__ == '__main__':
    n_data = len(glob.glob(image_dir + '\*.jpg'))   #the number of data


