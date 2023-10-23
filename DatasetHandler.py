"""import PIL
from sklearn.datasets import load_files

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split"""
import PIL
from keras.preprocessing import image
import os, sys
from os.path import join
from os import listdir
import numpy as np
"""
def load_images(root, min_side=299):
    all_imgs = []
    all_classes = []
    resize_count = 0
    invalid_count = 0
    for i, subdir in enumerate(listdir(root)):
        imgs = listdir(join(root, subdir))
        class_ix = class_to_ix[subdir]
        print(i, class_ix, subdir)
        for img_name in imgs:

            img_arr = img.imread(join(root, subdir, img_name))
            img_arr_rs = img_arr
            try:
                w, h, _ = img_arr.shape
                if w < min_side:
                    wpercent = (min_side/float(w))
                    hsize = int((float(h)*float(wpercent)))
                    #print('new dims:', min_side, hsize)
                    img_arr_rs = imresize(img_arr, (min_side, hsize))
                    resize_count += 1
                elif h < min_side:
                    hpercent = (min_side/float(h))
                    wsize = int((float(w)*float(hpercent)))
                    #print('new dims:', wsize, min_side)
                    img_arr_rs = imresize(img_arr, (wsize, min_side))
                    resize_count += 1
                all_imgs.append(img_arr_rs)
                all_classes.append(class_ix)
            except:
                print('Skipping bad image: ', subdir, img_name)
                invalid_count += 1
    print(len(all_imgs), 'images loaded')
    print(resize_count, 'images resized')
    print(invalid_count, 'images skipped')
    return np.array(all_imgs), np.array(all_classes)
    """
def load_images_formList(listImagesDir,high,width):
    all_imgs = []
    for img_name in listImagesDir:
        all_imgs.append(path_to_tensor(img_name,high,width))
    return np.vstack(all_imgs)


def path_to_tensor(img_path,high,width):
    img = image.load_img(img_path, target_size=(high, width))
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)
