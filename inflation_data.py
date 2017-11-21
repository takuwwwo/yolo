"""
inflate images for deep learning.
1. add noises.
2. rotate images by 4 directions.
"""

import os
import cv2
import glob
import numpy as np
import sys

image_dir = '.\image'
training_dir = '.\\training'

args = sys.argv
if len(args) == 3:
    image_dir = args[1]
    training_dir = args[2]
images = glob.glob(image_dir + '\*.jpg')


def rotateimage(im, angle):
    if len(im.shape) == 3:
        if angle == 90:
            return cv2.flip(np.transpose(im, [1, 0, 2]), 0)
        elif angle == 180:
            return cv2.flip(im, -1)
        else:
            return cv2.flip(np.transpose(im, [1, 0, 2]), 1)
    else:
        if angle == 90:
            return cv2.flip(im.T, 0)
        elif angle == 180:
            return cv2.flip(im, -1)
        else:
            return cv2.flip(im.T, 1)

print(len(images))
for path in images:
    image_path = os.path.splitext(path)[0]
    training_path = training_dir + '\\' + os.path.basename(path)

    print(training_path)
    if not(os.path.exists(training_path)):
        print(training_path)
        continue

    if os.path.exists(image_path + '_f0_i.jpg'):
        print(image_path + '_f0_i.jpg')
        continue

    image = cv2.imread(path, 0)
    train_image = cv2.imread(training_path)
    training_path = os.path.splitext(training_path)[0]

    row, col = image.shape
    image_size = row * col

    print(image_path)
    if image_path[-2:] == '_l':
        continue
    if image_path[-2:] == '_r':
        continue
    if image_path[-2:] == '_i':
        continue
    if image_path[-3:] == '_f0':
        continue
    if image_path[-3:] == '_f1':
        continue
    if image_path[-6:] == '_gauss':
        continue
    if image_path[-3:] == '_sp':
        continue

    """
    add gaussian noise
    """
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss_img = image + gauss
    gauss_img_path = image_path + '_gauss'
    gauss_train_path = training_path + '_gauss'

    """
    add salt and pepper noise
    """
    s_vs_p = 0.5  # ratio between salt and pepper.
    amount = 0.004
    cent = row * col // 2
    print(cent)
    sp_img = image.copy()
    # salt
    num_salt = np.ceil(amount * cent * s_vs_p)
    coords_row = np.random.randint(0, row-1, num_salt)
    coords_col = np.random.randint(0, col-1, num_salt)
    print(len(coords_row))
    sp_img[coords_row, coords_col] = 255

    # pepper
    num_pepper = np.ceil(amount * cent * s_vs_p)
    coords_row = np.random.randint(0, row-1, num_pepper)
    coords_col = np.random.randint(0, col-1, num_pepper)
    sp_img[coords_row, coords_col] = 0

    sp_img_path = image_path + '_sp'
    sp_train_path = training_path + '_sp'

    # write gauss and salt_pepper images
    cv2.imwrite(gauss_img_path + '.jpg', gauss_img)
    cv2.imwrite(sp_img_path + '.jpg', sp_img)
    cv2.imwrite(gauss_train_path + '.jpg', train_image)
    cv2.imwrite(sp_train_path + '.jpg', train_image)

    # write flipped image
    cv2.imwrite(image_path + '_f0.jpg', cv2.flip(image, 0))
    cv2.imwrite(gauss_img_path + '_f0.jpg', cv2.flip(gauss_img, 0))
    cv2.imwrite(sp_img_path + '_f0.jpg', cv2.flip(sp_img, 0))
    cv2.imwrite(training_path + '_f0.jpg', cv2.flip(train_image, 0))
    cv2.imwrite(gauss_train_path + '_f0.jpg', cv2.flip(train_image, 0))
    cv2.imwrite(sp_train_path + '_f0.jpg', cv2.flip(train_image, 0))
    """
    cv2.imwrite(image_path + '_f1.jpg', cv2.flip(image, 1))
    cv2.imwrite(gauss_img_path + '_f1.jpg', cv2.flip(gauss_img, 1))
    cv2.imwrite(sp_img_path + '_f1.jpg', cv2.flip(sp_img, 1))
    cv2.imwrite(training_path + '_f1.jpg', cv2.flip(train_image, 1))
    cv2.imwrite(gauss_train_path + '_f1.jpg', cv2.flip(train_image, 1))
    cv2.imwrite(sp_train_path + '_f1.jpg', cv2.flip(train_image, 1))
    """

    # write rotated image
    # left
    cv2.imwrite(image_path + '_l.jpg', rotateimage(image, 90))
    cv2.imwrite(gauss_img_path + '_l.jpg', rotateimage(gauss_img, 90))
    cv2.imwrite(sp_img_path + '_l.jpg', rotateimage(sp_img, 90))
    cv2.imwrite(training_path + '_l.jpg', rotateimage(train_image, 90))
    cv2.imwrite(gauss_train_path + '_l.jpg', rotateimage(train_image, 90))
    cv2.imwrite(sp_train_path + '_l.jpg', rotateimage(train_image, 90))
    cv2.imwrite(image_path + '_f0_l.jpg', rotateimage(cv2.flip(image, 0), 90))
    cv2.imwrite(gauss_img_path + '_f0_l.jpg', rotateimage(cv2.flip(gauss_img, 0), 90))
    cv2.imwrite(sp_img_path + '_f0_l.jpg', rotateimage(cv2.flip(sp_img, 0), 90))
    cv2.imwrite(training_path + '_f0_l.jpg', rotateimage(cv2.flip(train_image, 0), 90))
    cv2.imwrite(gauss_train_path + '_f0_l.jpg', rotateimage(cv2.flip(train_image, 0), 90))
    cv2.imwrite(sp_train_path + '_f0_l.jpg', rotateimage(cv2.flip(train_image, 0), 90))
    """
    cv2.imwrite(image_path + '_f1_l.jpg', rotateimage(cv2.flip(image, 1), 90))
    cv2.imwrite(gauss_img_path + '_f1_l.jpg', rotateimage(cv2.flip(gauss_img, 1), 90))
    cv2.imwrite(sp_img_path + '_f1_l.jpg', rotateimage(cv2.flip(sp_img, 1), 90))
    cv2.imwrite(training_path + '_f1_l.jpg', rotateimage(cv2.flip(train_image, 1), 90))
    cv2.imwrite(gauss_train_path + '_f1_l.jpg', rotateimage(cv2.flip(train_image, 1), 90))
    cv2.imwrite(sp_train_path + '_f1_l.jpg', rotateimage(cv2.flip(train_image, 1), 90))
    """

    # inverse
    cv2.imwrite(image_path + '_i.jpg', rotateimage(image, 180))
    cv2.imwrite(gauss_img_path + '_i.jpg', rotateimage(gauss_img, 180))
    cv2.imwrite(sp_img_path + '_i.jpg', rotateimage(sp_img, 180))
    cv2.imwrite(training_path + '_i.jpg', rotateimage(train_image, 180))
    cv2.imwrite(gauss_train_path + '_i.jpg', rotateimage(train_image, 180))
    cv2.imwrite(sp_train_path + '_i.jpg', rotateimage(train_image, 180))
    cv2.imwrite(image_path + '_f0_i.jpg', rotateimage(cv2.flip(image, 0), 180))
    cv2.imwrite(gauss_img_path + '_f0_i.jpg', rotateimage(cv2.flip(gauss_img, 0), 180))
    cv2.imwrite(sp_img_path + '_f0_i.jpg', rotateimage(cv2.flip(sp_img, 0), 180))
    cv2.imwrite(training_path + '_f0_i.jpg', rotateimage(cv2.flip(train_image, 0), 180))
    cv2.imwrite(gauss_train_path + '_f0_i.jpg', rotateimage(cv2.flip(train_image, 0), 180))
    cv2.imwrite(sp_train_path + '_f0_i.jpg', rotateimage(cv2.flip(train_image, 0), 180))
    """
    cv2.imwrite(image_path + '_f1_i.jpg', rotateimage(cv2.flip(image, 1), 180))
    cv2.imwrite(gauss_img_path + '_f1_i.jpg', rotateimage(cv2.flip(gauss_img, 1), 180))
    cv2.imwrite(sp_img_path + '_f1_i.jpg', rotateimage(cv2.flip(sp_img, 1), 180))
    cv2.imwrite(training_path + '_f1_i.jpg', rotateimage(cv2.flip(train_image, 1), 180))
    cv2.imwrite(gauss_train_path + '_f1_i.jpg', rotateimage(cv2.flip(train_image, 1), 180))
    cv2.imwrite(sp_train_path + '_f1_i.jpg', rotateimage(cv2.flip(train_image, 1), 180))
    """

    # right
    cv2.imwrite(image_path + '_r.jpg', rotateimage(image, 270))
    cv2.imwrite(gauss_img_path + '_r.jpg', rotateimage(gauss_img, 270))
    cv2.imwrite(sp_img_path + '_r.jpg', rotateimage(sp_img, 270))
    cv2.imwrite(training_path + '_r.jpg', rotateimage(train_image, 270))
    cv2.imwrite(gauss_train_path + '_r.jpg', rotateimage(train_image, 270))
    cv2.imwrite(sp_train_path + '_r.jpg', rotateimage(train_image, 270))
    cv2.imwrite(image_path + '_f0_r.jpg', rotateimage(cv2.flip(image, 0), 270))
    cv2.imwrite(gauss_img_path + '_f0_r.jpg', rotateimage(cv2.flip(gauss_img, 0), 270))
    cv2.imwrite(sp_img_path + '_f0_r.jpg', rotateimage(cv2.flip(sp_img, 0), 270))
    cv2.imwrite(training_path + '_f0_r.jpg', rotateimage(cv2.flip(train_image, 0), 270))
    cv2.imwrite(gauss_train_path + '_f0_r.jpg', rotateimage(cv2.flip(train_image, 0), 270))
    cv2.imwrite(sp_train_path + '_f0_r.jpg', rotateimage(cv2.flip(train_image, 0), 270))
    """
    cv2.imwrite(image_path + '_f1_r.jpg', rotateimage(cv2.flip(image, 1), 270))
    cv2.imwrite(gauss_img_path + '_f1_r.jpg', rotateimage(cv2.flip(gauss_img, 1), 270))
    cv2.imwrite(sp_img_path + '_f1_r.jpg', rotateimage(cv2.flip(sp_img, 1), 270))
    cv2.imwrite(training_path + '_f1_r.jpg', rotateimage(cv2.flip(train_image, 1), 270))
    cv2.imwrite(gauss_train_path + '_f1_r.jpg', rotateimage(cv2.flip(train_image, 1), 270))
    cv2.imwrite(sp_train_path + '_f1_r.jpg', rotateimage(cv2.flip(train_image, 1), 270))
    """
