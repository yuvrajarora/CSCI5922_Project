from skimage.transform import rescale
from skimage import img_as_uint
import skimage.io as io
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as trans
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



def rescale_img(img_path, img_modified_path, num_imgs, x_factor, y_factor):

    for i in range(0, num_imgs):
        path = str(i) + '.tif'
        img = io.imread(img_path + path)
        img_resized = rescale(img, (x_factor, y_factor), anti_aliasing=False)
        img_resized = img_as_uint(img_resized)
        path = str(i) + '.tif'
        io.imsave(img_modified_path + path, img_resized)


def trainGenerator(batch_size, train_path, img_folder, mask_folder, aug_dict, color_mode='grayscale',
                   mask_color_mode='grayscale', target_size=(576, 576), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(train_path, classes=[img_folder], class_mode=None,
                                                        color_mode=color_mode, target_size=target_size,
                                                        batch_size=batch_size, seed=seed)

    mask_generator = mask_datagen.flow_from_directory(train_path, classes=[mask_folder], class_mode=None,
                                                      color_mode=mask_color_mode, target_size=target_size,
                                                      batch_size=batch_size, seed=seed)

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)
        yield (img, mask)


def adjustData(img,mask):
    if(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def testGenerator(test_path,num_image,target_size = (576,576),as_gray = True):
    for i in range(0, num_image):
        img = io.imread(os.path.join(test_path,"%d.tif"%i),as_gray = as_gray)
        img = img / 255
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img

