from Unet_Model import *
from utility import *
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
import skimage.io as io
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


num_train_imgs = 65
num_train_labels = 65
num_test_imgs = 110

x_factor = 0.835
y_factor = 0.917
x_factor_test = 0.8022
y_factor_test = 0.873

rescale_img('train/image/', 'train/image_processed/',num_train_imgs, x_factor, y_factor)
rescale_img('train/label/', 'train/label_processed/', num_train_labels, x_factor, y_factor)
rescale_img('test/image/', 'test/image_processed/', num_test_imgs, x_factor_test, y_factor_test)

data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
                     zoom_range=0.05, horizontal_flip=True, fill_mode='nearest')


batch_size = 2
IMG_HEIGHT = 576
IMG_WIDTH = 576
CHANNEL = 1
IMAGE_PATH = 'image_processed'
LABEL_PATH = 'label_processed'
MAIN_FOLDER_PATH = 'train'

myGene = trainGenerator(batch_size, MAIN_FOLDER_PATH, IMAGE_PATH, LABEL_PATH,data_gen_args)
model = unet_model(IMG_HEIGHT,IMG_WIDTH,CHANNEL)
model.fit_generator(myGene,steps_per_epoch=300,epochs=25,callbacks=None)
model.save('unet_model.h5')

TEST_IMG_PATH = 'test/image_processed/'
model = load_model('unet_model.h5')
testGene = testGenerator(TEST_IMG_PATH, num_test_imgs)
results = model.predict_generator(testGene, num_test_imgs, verbose=1)

for i in range(num_test_imgs):
    cmap = plt.cm.Greys
    norm = plt.Normalize(vmin=0, vmax=0.7)
    image = np.squeeze(cmap(norm(np.multiply(results[i], 255))))
    plt.imsave("test/results/" + str(i) + '.tif', image)