import argparse
import os
import numpy as np
import cv2

import keras
from keras.models import load_model
import h5py
from keras import __version__ as keras_version
from keras.models import Sequential
from keras.layers import Dense, Cropping2D, Lambda, Flatten, Dropout, Convolution2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from vis.visualization import visualize_cam, overlay
from vis.utils import utils

def readImageRGB(path):
    #imread returns BGR
    return np.flip(cv2.imread(path), 2)

def createModel(kerasVer, rows, cols, chs):
    model = Sequential()

    #160x320x3
    model.add(Cropping2D(cropping=((57,25), (1,1)), data_format='channels_last', input_shape=(rows, cols, chs)))
    #78x318x3
    model.add(Lambda(lambda x: x/127.5-1.0))
    model.add(Conv2D(16, (3, 3), activation='relu', bias=False))
    #76x316x16
    model.add(MaxPooling2D((2,2)))
    #38x158x16
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(32, (3, 3), activation='relu', bias=False))
    #36x156x32
    model.add(MaxPooling2D((2,2)))
    #18x78x32
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (3, 3), activation='relu', bias=False))
    #16x76x64
    model.add(MaxPooling2D((2,2)))
    #8x38x64
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3, 3), activation='relu', bias=False))
    #6x36x128
    model.add(MaxPooling2D((2,2)))
    #3x18x128
    model.add(BatchNormalization(axis=3))
    model.add(Convolution2D(256, 3, 3, activation='relu', bias=False))
    #1x16x256
    model.add(Flatten())
    #4092
    model.add(BatchNormalization(axis=1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    #256
    model.add(BatchNormalization(axis=1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    #32
    model.add(Dense(1))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization of steering decisions')
    parser.add_argument('-m', '--model', type=str, default='model.h5', required=True, dest='model')
    parser.add_argument('-i', '--inpfolder', type=str, required=True, dest='image_folder')
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version, ', but the model was built using ', model_version)

    model = createModel(keras_version, 160, 320, 3)
    model.load_weights(args.model)
    model.compile(loss='mse', optimizer='adam')

    if not os.path.exists(args.image_folder):
        print('Input folder not found')
        exit(1)

    fig = plt.figure()
    plt.axis('off')
    ax = fig.add_subplot(111)

    outFolder = args.image_folder + '_out'

    filelist = os.listdir(args.image_folder)
    i=0

    for imgpath in filelist:
        if not imgpath.endswith(".jpg"):
            continue

        img = readImageRGB(args.image_folder + '/' + imgpath)

        heatmapLeft = visualize_cam(model, -1, filter_indices=0, seed_input=img, penultimate_layer_idx=12, grad_modifier='negate')
        heatmapRight = visualize_cam(model, -1, filter_indices=0, seed_input=img, penultimate_layer_idx=12, grad_modifier=None)

        #input is 160x320x3 with cropping layer (57,25),(1,1), the heatmap does not account for that
        heatmapLeft = cv2.resize(heatmapLeft, (320, 78))
        heatmapRight = cv2.resize(heatmapRight, (320, 78))

        heatmapLeft = np.concatenate((np.zeros((57,320,3)), heatmapLeft, np.zeros((25,320,3))))
        heatmapRight = np.concatenate((np.zeros((57,320,3)), heatmapRight, np.zeros((25,320,3))))

        #heatmapLeft = 255 * np.delete(cmap(heatmapLeft), 3, 2)
        #heatmapRight = 255 * np.delete(cmap(heatmapRight), 3, 2)

        outLeft = overlay(img, heatmapLeft, alpha=0.7)
        outRight = overlay(img, heatmapRight, alpha=0.7)

        out = np.concatenate((outLeft, outRight))

        if i==0:
            if not os.path.isdir(outFolder):
                os.makedirs(outFolder)

        ax.imshow(out)
        fig.savefig(outFolder + '/' +  imgpath)

        print('Saved {}/{} - "{}"'.format(i, len(filelist), imgpath))
        i += 1

