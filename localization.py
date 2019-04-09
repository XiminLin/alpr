from skimage import measure
from skimage.transform import resize
from skimage.filters import threshold_otsu, threshold_local, threshold_isodata
from skimage.color import rgb2gray
import os
import re
import gc
import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
## keras
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D



plot_direc = 'plot'
data_direc = 'us'
shape = (720, 1280, 3) ## better to change
# nimg = 10 ## debug
nimg = 222 


def read_data():
    allimage = glob.glob(data_direc + '/*.jpg')
    image = np.empty(nimg, dtype=object)
    label = np.empty(nimg, dtype=object)
    for i, image_name in enumerate(allimage):
        label_name = image_name.rsplit('.')[0] + ".txt"
        img = plt.imread(image_name)
        image[i] = img
        with open(label_name, 'r') as f:
            txt = f.read()
        label[i] = txt

        ## debug
        if i == nimg-1: break

    return image, label

## reshape to shape
def reshape_image_label(image, label):

    image_reshape = np.empty((image.shape[0], shape[0], shape[1], shape[2]))
    for i, img in enumerate(image):
        image_reshape[i,:,:,:] = resize(img, output_shape=shape, mode='reflect',  anti_aliasing=True)
    return image_reshape



def gray_2_binary(image):
    bin_image = np.empty(image.shape)
    for i, img in enumerate(image):
        thresh_val = threshold_otsu(img)
        bin_image[i,:,:] = img > thresh_val
    return bin_image

## get bounding box
## (x,y,width,height)
def get_bb(lab):
    lab = lab.strip('\n').split('\t')
    bb = np.array(lab[1:5], dtype='int')
    return bb

def get_model():
    pass
    

if __name__ == '__main__':

    image, label = read_data()




    # batch_size = 32
    # num_classes = 10
    # epochs = 100
    # data_augmentation = True
    # num_predictions = 20
    # save_dir = os.path.join(os.getcwd(), 'saved_models')
    # model_name = 'localization_trained_model.h5'






