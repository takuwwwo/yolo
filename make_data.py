import cv2
import glob
import sys
import numpy as np
import os
import math
import network


args = sys.argv
image_size = network.IMAGE_SIZE
S = network.S
B = network.B
C = network.C
last_output = network.last_output

# check sys.argv points directory correctly.
if len(args) != 5:
    print("make_data.py : args is not proper.")
    quit()

# directory for images for training. Fill rectangles with red color.
training_dir = args[1]

# directory for original images
image_dir = args[2]

# directory for storing label data
label_dir = args[3]

# directory for storing images for checking whether labels are correct.
check_dir = args[4]

# image data and labeldata
imdata = glob.glob(training_dir + '\*.jpg')
lbdata = glob.glob(label_dir + '\*.npy')

# making label starts
for training_path in imdata:
    label_path = label_dir + '\\' + os.path.basename(training_path[:-4]) + '.npy'     # remove '.jpg' and add '.npy'
    if label_path in lbdata:     # if label_data exists, need not to make label_data
        continue

    image = cv2.imread(training_path, 1)          # read image by 3 channel coloring type
    if not(os.path.exists(image_dir + '\\' + os.path.basename(training_path))):
        print("no_image : " + training_path)
        continue

    print("exist " + training_path)
    image_org = cv2.imread(image_dir + '\\' + os.path.basename(training_path), 0)
    compress = image_size / image.shape[0], image_size / image.shape[1]
    image = image.astype(np.int16)    # for substract each other.
    x = (image[:, :, 2] - image[:, :, 1] - image[:, :, 0] > 100) * 255
    x = x.astype(np.uint8)    # undo type.

    # cv2.imwrite(check_dir + '\\' + os.path.basename(training_path), x)
    x, contours, hierarchy = cv2.findContours(x, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for contour in contours:
        points = []
        xmin = min(contour, key=lambda t: t[0][0])[0][0]
        ymin = min(contour, key=lambda t: t[0][1])[0][1]
        xmax = max(contour, key=lambda t: t[0][0])[0][0]
        ymax = max(contour, key=lambda t: t[0][1])[0][1]

        center = np.array(((xmax + xmin) / 2, (ymax + ymin) / 2))
        height = xmax - xmin + 0.0
        width = ymax - ymin + 0.0

        # by multipling compress, adjust label value to image_size(448)
        center[0] *= compress[1]; center[1] *= compress[0]
        height *= compress[1]; width *= compress[0]

        results.append((center, height, width))

    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)      # resize
    # print(image_org.shape, image_size)
    image_org = cv2.resize(image_org, (image_size, image_size), interpolation=cv2.INTER_AREA)
    # results : ((448, 448), (center, height, width))
    """
    for c, h, w in results:
        p1 = (math.floor(c[0] - h/2), math.floor(c[1] - w/2))
        p2 = (math.floor(c[0] - h/2), math.ceil(c[1] + w/2))
        p3 = (math.ceil(c[0] + h/2), math.floor(c[1] - w/2))
        p4 = (math.ceil(c[0] + h/2), math.ceil(c[1] + w/2))
        cv2.line(image, p1, p2, (0, 255, 0), 2)
        cv2.line(image, p1, p3, (0, 255, 0), 2)
        cv2.line(image, p2, p4, (0, 255, 0), 2)
        cv2.line(image, p3, p4, (0, 255, 0), 2)

        cv2.imwrite(check_dir + '\\' + os.path.basename(training_path), image)
    """

    label = np.zeros((S, S, last_output))
    for c, h, w in results:
        comp = image_size / S
        x = math.floor(c[0] / comp)
        y = math.floor(c[1] / comp)
        label[x][y][0] = (c[0] - comp * x) / comp
        label[x][y][1] = (c[1] - comp * y) / comp
        label[x][y][2] = w / image_size
        label[x][y][3] = h / image_size
        label[x][y][4] = 1.0

    """
    for i in range(S):
        for j in range(S):
            if label[i][j][4] == 1.0:
                x, y = label[i][j][0] * comp + comp * i, label[i][j][1] * comp + comp * j
                w, h = label[i][j][2] * image_size, label[i][j][3] * image_size
                p1 = (math.floor(x - h / 2), math.floor(y - w / 2))
                p2 = (math.floor(x - h / 2), math.ceil(y + w / 2))
                p3 = (math.ceil(x + h / 2), math.floor(y - w / 2))
                p4 = (math.ceil(x + h / 2), math.ceil(y + w / 2))
                cv2.line(image, p1, p2, (0, 255, 0), 2)
                cv2.line(image, p1, p3, (0, 255, 0), 2)
                cv2.line(image, p2, p4, (0, 255, 0), 2)
                cv2.line(image, p3, p4, (0, 255, 0), 2)
    """

    # cv2.imwrite(check_dir + '\\' + os.path.basename(training_path), image)
    # cv2.imwrite(check_dir + '\\org_' + os.path.basename(training_path), image_org)
    np.save(label_path, (image_org, label))
