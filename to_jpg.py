import os
import glob
import sys

if len(sys.argv) == 2:
    image_dir = sys.argv[1]
else:
    image_dir = ".\image"

"""
GIF, TIF 
PNG
"""
images_path = []
images_path.extend(glob.glob(image_dir+'\*.gif'))
images_path.extend(glob.glob(image_dir+'\*.tif'))
images_path.extend(glob.glob(image_dir+'\*.eps'))
images_path.extend(glob.glob(image_dir+'\*.png'))

for path in images_path:
    filename = os.path.splitext(path)[0] + '.jpg'
    print("filename" + filename)
    if os.system("magick convert -density 300 " + path + " " + filename) == 0:
        print(path + ' is converted correctly.')
    else:
        print(path + ' cannot be converted.')


