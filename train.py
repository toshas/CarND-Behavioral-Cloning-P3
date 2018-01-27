import os
import csv
import argparse
import cv2
import numpy as np
import pandas as pd
import sklearn
from distutils.version import StrictVersion
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#AWS-friendly plotting
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import keras
from keras.models import Sequential
from keras.layers import Dense, Cropping2D, Lambda, Flatten, Dropout, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint

DEF_DATA_DIR = './data'
SIDEVIEW_DELTA_ANGLE = 0.2

def readImageRGB(path):
    #imread returns BGR
    return np.flip(cv2.imread(path), 2)

def trimSimulatorFilepath(path):
    parts = path.split('/')[-2:]
    return parts[0] + '/' + parts[1]

def batchGenerator(samples, bAugment, bUseRear, batchSz, datadir):
    nSamples = len(samples)

    if bAugment and bUseRear:
        batchSz //= 6
    elif bAugment:
        batchSz //= 2
    assert batchSz != 0

    while 1:
        shuffle(samples)
        for offset in range(0, nSamples, batchSz):
            batchSamples = samples[offset : offset+batchSz]

            images = []
            angles = []
            for batchSample in batchSamples:
                pathCenter = datadir + '/' + trimSimulatorFilepath(batchSample[0])
                if not os.path.exists(pathCenter):
                    raise IOError('Inconsistent dataset, {} is missing'.format(pathCenter))

                imgCenter = readImageRGB(pathCenter)
                angleCenter = float(batchSample[3])

                images.append(imgCenter)
                angles.append(angleCenter)

                # augmentations follow
                if bAugment:
                    images.append(np.fliplr(imgCenter))
                    angles.append(-angleCenter)

                    if bUseRear:
                        imgLeft = readImageRGB(datadir + '/' + trimSimulatorFilepath(batchSample[1]))
                        imgRight = readImageRGB(datadir + '/' + trimSimulatorFilepath(batchSample[2]))

                        angleLeft = angleCenter + SIDEVIEW_DELTA_ANGLE
                        angleRight = angleCenter - SIDEVIEW_DELTA_ANGLE

                        images.append(imgLeft)
                        angles.append(angleLeft)

                        images.append(np.fliplr(imgLeft))
                        angles.append(-angleLeft)

                        images.append(imgRight)
                        angles.append(angleRight)

                        images.append(np.fliplr(imgRight))
                        angles.append(-angleRight)

            X = np.array(images)
            y = np.array(angles)
            yield X, y

def reconcileDataset(samples, datadir):
    cleanSamples = []
    for sample in samples:
        pathCenter = datadir + '/' + trimSimulatorFilepath(sample[0])
        if not os.path.exists(pathCenter):
            print('Image for entry {} not found, skipping'.format(pathCenter))
            continue
        cleanSamples.append(sample)
    return cleanSamples

def openTelemetry(datadir):
    filename = datadir + '/driving_log.csv'
    if not os.path.exists(filename):
        raise IOError('File could not be opened "{}"'.format(filename))
    samples = pd.read_csv(filename, sep='\s*,\s*', engine='python').values.tolist()
    return reconcileDataset(samples, datadir)

def openTelemetryTrainValidFromSingleDataset(datadir):
    samples = openTelemetry(datadir)
    trainSamples, validSamples = train_test_split(samples, test_size=0.2)
    return trainSamples, validSamples

def openTelemetryTrainValidFromDatasets(dirTrain, dirValid):
    return openTelemetry(dirTrain), openTelemetry(dirValid)

def createModel(kerasVer, rows, cols, chs):
    model = Sequential()

    #160x320x3
    if StrictVersion(kerasVer) <= StrictVersion('1.2.1'):
        model.add(Cropping2D(cropping=((57,25), (1,1)), dim_ordering='tf', input_shape=(rows, cols, chs)))
    else:
        model.add(Cropping2D(cropping=((57,25), (1,1)), data_format='channels_last', input_shape=(rows, cols, chs)))
    #78x318x3
    model.add(Lambda(lambda x: x/127.5-1.0))
    model.add(Convolution2D(16, 3, 3, activation='relu', bias=False))
    #76x316x16
    model.add(MaxPooling2D((2,2)))
    #38x158x16
    model.add(BatchNormalization(axis=3))
    model.add(Convolution2D(32, 3, 3, activation='relu', bias=False))
    #36x156x32
    model.add(MaxPooling2D((2,2)))
    #18x78x32
    model.add(BatchNormalization(axis=3))
    model.add(Convolution2D(64, 3, 3, activation='relu', bias=False))
    #16x76x64
    model.add(MaxPooling2D((2,2)))
    #8x38x64
    model.add(BatchNormalization(axis=3))
    model.add(Convolution2D(128, 3, 3, activation='relu', bias=False))
    #6x36x128
    model.add(MaxPooling2D((2,2)))
    #3x18x128
    model.add(BatchNormalization(axis=3))
    model.add(Convolution2D(256, 3, 3, activation='relu', bias=False))
    #1x16x256
    model.add(Flatten())
    #4092
    model.add(BatchNormalization(mode=1, axis=1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    #256
    model.add(BatchNormalization(mode=1, axis=1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    #32
    model.add(Dense(1))
    return model

def train(dirTrain, dirValid, batchSz, epochs, bResume=False):
    if dirValid is None:
        telemetryTrainSamples, telemetryValidSamples = openTelemetryTrainValidFromSingleDataset(dirTrain)
        dirValid = dirTrain
    else:
        telemetryTrainSamples, telemetryValidSamples = openTelemetryTrainValidFromDatasets(dirTrain, dirValid)

    print('Training with {} training and {} validation samples, batch size {}, epochs {}'.format(len(telemetryTrainSamples), len(telemetryValidSamples), batchSz, epochs))
    assert len(telemetryTrainSamples) > 0 and len(telemetryValidSamples) > 0

    bAugment = True
    bUseRear = False
    trainGenerator = batchGenerator(telemetryTrainSamples, bAugment, bUseRear, batchSz, dirTrain)
    validGenerator = batchGenerator(telemetryValidSamples, False, False, batchSz, dirValid)

    #TODO: remove these hardcodings, use tee(trainGenerator) and read the first image to get shape
    rows, cols, chs = 160, 320, 3

    kerasVer = keras.__version__
    print('Keras version is {}'.format(keras.__version__))

    model = createModel(kerasVer, rows, cols, chs)

    modelFilename = 'model.h5'
    if os.path.exists(modelFilename) and bResume:
        model.load_weights(modelFilename)
        print('Loaded model from file')

    model.compile(loss='mse', optimizer='adam')

    samplesPerEpoch = len(telemetryTrainSamples)
    if bAugment:
        samplesPerEpoch *= 2
        if bUseRear:
            samplesPerEpoch *= 3

    checkpointer = ModelCheckpoint(filepath=modelFilename, verbose=1, save_best_only=True)
    report = model.fit_generator(trainGenerator, samples_per_epoch=samplesPerEpoch, validation_data=validGenerator, \
                    nb_val_samples=len(telemetryValidSamples), nb_epoch=epochs, callbacks=[checkpointer])

    plt.plot(report.history['loss'])
    plt.plot(report.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('loss.png')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', type=str, default=DEF_DATA_DIR, required=True, dest='train', help='Dir with a CSV capture and images from the simulator, used for training')
parser.add_argument('-v', '--valid', type=str, default=None, required=False, dest='valid', help='Dir with a CSV capture and images from the simulator, used for validation (optional)')
parser.add_argument('-b', '--batch', type=int, default=36, dest='batch', help='Batch size')
parser.add_argument('-e', '--epochs', type=int, default=3, dest='epochs', help='Number of epochs')
args = parser.parse_args()

with keras.backend.get_session():
    train(args.train, args.valid, args.batch, args.epochs)

