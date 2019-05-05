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
def reshape_image(image):
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

## draw one image with the bounding box of license plate
def draw_bb(img, lab):
    bb = get_bb(lab)
    fig, ax = plt.subplots()
    ax.imshow(img)
    rect = patches.Rectangle((bb[0],bb[1]), bb[2], bb[3],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()

def testing_dataset(image, label):
    ## get all bounding box information
    all_shape = np.empty((nimg, 2), dtype='int')
    all_bb = np.empty((nimg, 4), dtype='int')
    for i, img in enumerate(image):
        all_shape[i,:] = np.array(img.shape[:-1])
        bb = get_bb(label[i])
        all_bb[i,:] = bb

    ## check the range of the license position
    x_position = all_bb[:,0].astype('float') / all_shape[:,1].astype('float')
    y_position = all_bb[:,1].astype('float') / all_shape[:,0].astype('float')

    ax = sns.distplot(y_position, bins=20, kde=False)
    ax.set_title('y_position')
    plt.savefig(plot_direc + '/y_position.jpg')
    plt.figure()
    ax = sns.distplot(x_position, bins=20, kde=False)
    ax.set_title('x_position')
    plt.savefig(plot_direc + '/x_position.jpg')
    plt.figure()

    ## check the size of the license plate relative to the image
    width = all_bb[:,2].astype('float') / all_shape[:,0].astype('float')
    height = all_bb[:,3].astype('float') / all_shape[:,1].astype('float')

    ax = sns.distplot(width, bins=20, kde=False)
    ax.set_title('width')
    plt.savefig(plot_direc + '/width.jpg')
    plt.figure()
    ax = sns.distplot(height, bins=20, kde=False)
    ax.set_title('height')
    plt.savefig(plot_direc + '/height.jpg')

    plt.show()

if __name__ == '__main__':
    sns.set()
    image, label = read_data()
    # image = reshape_image(image)

    ## rgb 2 grayscale
    # gray_image = rgb2gray(image)

    ## grayscale to binary
    # bin_image = gray_2_binary(gray_image)

    ## draw the single image
    # test_img = bin_image[0,:,:]
    # draw_bb(test_img, label[0])

    testing_dataset(image, label)


