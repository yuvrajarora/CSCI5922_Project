
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from skimage.transform import rescale
from skimage import img_as_uint,img_as_ubyte
import tifffile as tiff

from skimage.restoration import denoise_wavelet
import tensorflow as tf
import skimage.io as io
from skimage import util 
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,ConvLSTM2D
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv3DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D, MaxPooling3D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt


#Creating Conv2DLSTM model

seq = Sequential()
filter1 = 40
seq.add(ConvLSTM2D(filters=filter1, kernel_size=(3, 3),
                   input_shape=(None,512,512, 1),
                   padding='same',data_format='channels_last' , return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=filter1, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=filter1, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=filter1, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])


#Utility functions
def trainGenerator(batch_size, train_path, img_folder, mask_folder, aug_dict, color_mode = 'grayscale', 
                   mask_color_mode = 'grayscale', target_size = (512,512), seed = 1):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(train_path, classes = [img_folder], class_mode = None,
                                                        color_mode = color_mode, target_size = target_size, 
                                                        batch_size = batch_size, seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(train_path, classes = [mask_folder], class_mode = None,
                                                      color_mode = mask_color_mode, target_size = target_size,
                                                      batch_size = batch_size, seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (np.expand_dims(np.expand_dims(img[0],axis=0),axis=0),np.expand_dims(np.expand_dims(mask[0],axis=0),axis=0))
		
		
def adjustData(img,mask):
    j = 0
    if(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def testGenerator(test_path,num_image,target_size = (512,512)):#,as_gray = True):
    for i in range(0, num_image):
        img = io.imread(os.path.join(test_path,"%d.tif"%i))#,as_gray = as_gray)
        #img = img / 255
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield (np.expand_dims(np.expand_dims(img[0],axis=0),axis=0))
		


data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
                     zoom_range=0.05, horizontal_flip=True, fill_mode='nearest')

myGene = trainGenerator(1,'train','image_processed','label_processed',data_gen_args)
seq.fit_generator(myGene,steps_per_epoch=130,epochs=1,callbacks=None,verbose=1)


testGene = testGenerator("test/image_processed/", 3)
results = seq.predict_generator(testGene,3,verbose=1)
plt.imshow(util.invert(results[2][0]).reshape(512,512),cmap='Greys')
plt.imshow(util.invert(results[0][0]).reshape(512,512),cmap='Greys')
seq.save('convlstm2d_model.h5')

TEST_IMG_PATH = 'test/image_processed/'
num_test_imgs = 110
model = load_model('convlstm2d_model.h5')
testGene = testGenerator(TEST_IMG_PATH, num_test_imgs)
results = model.predict_generator(testGene, num_test_imgs, verbose=1)

for i in range(num_test_imgs):
    cmap = plt.cm.Greys
    plt.imsave("test/results/" + str(i) + '.tif', cmap(util.invert(results[i][0].reshape(512,512))))

	
#U-Net with Conv2DLSTM

def unet_model (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    
    inputs = Input((None,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    c1 = ConvLSTM2D(64, (3, 3), activation='elu', data_format="channels_last", 
                    kernel_initializer='he_normal', padding='same', return_sequences=True) (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = ConvLSTM2D(64, (3, 3), activation='elu', data_format="channels_last", 
                    kernel_initializer='he_normal', padding='same', return_sequences=True) (c1)
    p1 = MaxPooling3D((2,2, 2)) (c1)

    c2 = ConvLSTM2D(128, (3, 3), activation='elu', data_format="channels_last", 
                    kernel_initializer='he_normal', padding='same', return_sequences=True) (p1)
    c2 = Dropout(0.1) (c2)
    c2 = ConvLSTM2D(128, (3, 3), activation='elu', data_format="channels_last", 
                    kernel_initializer='he_normal', padding='same', return_sequences=True) (c2)
    p2 = MaxPooling3D((2,2, 2)) (c2)

    c3 = ConvLSTM2D(256, (3, 3), activation='elu', data_format="channels_last", 
                    kernel_initializer='he_normal', padding='same', return_sequences=True) (p2)
    c3 = Dropout(0.2) (c3)
    c3 = ConvLSTM2D(256, (3, 3), activation='elu', data_format="channels_last", 
                    kernel_initializer='he_normal', padding='same', return_sequences=True) (c3)
    p3 = MaxPooling3D((2,2, 2)) (c3)

    c4 = ConvLSTM2D(512, (3, 3), activation='elu', data_format="channels_last", 
                    kernel_initializer='he_normal', padding='same', return_sequences=True) (p3)
    c4 = Dropout(0.2) (c4)
    c4 = ConvLSTM2D(512, (3, 3), activation='elu', data_format="channels_last", 
                    kernel_initializer='he_normal', padding='same', return_sequences=True) (c4)
    p4 = MaxPooling3D((2,2, 2)) (c4)

    c5 = ConvLSTM2D(1024, (3, 3), activation='elu', data_format="channels_last", 
                    kernel_initializer='he_normal', padding='same', return_sequences=True) (p4)
    c5 = Dropout(0.3) (c5)
    c5 = ConvLSTM2D(1024, (3, 3), activation='elu', data_format="channels_last", 
                    kernel_initializer='he_normal', padding='same', return_sequences=True) (c5)

    u6 = Conv3DTranspose(512, (2, 2,2), strides=(2, 2,2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(512, (3,3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv3D(512, (3,3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv3DTranspose(256, (2,2, 2), strides=(2,2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(256, (3,3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv3D(256, (3,3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv3DTranspose(128, (2,2, 2), strides=(2,2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(128, (3,3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv3D(128, (3,3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv3DTranspose(64, (2,2, 2), strides=(2,2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv3D(64, (3,3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv3D(64, (3,3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv3D(1, (1,1, 1), activation='sigmoid', padding='same') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model