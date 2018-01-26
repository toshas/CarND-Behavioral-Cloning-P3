import os
import csv
from distutils.version import StrictVersion
import cv2
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import argparse

import keras
from keras.models import Sequential
from keras.layers import Dense, Cropping2D, Lambda, Flatten
from keras.callbacks import ModelCheckpoint

DEF_DATA_DIR = './data'
SIDEVIEW_DELTA_ANGLE = 0.2
AUG_MULTIPLIER = 6

def batchGenerator(samples, bAugment, batchSz, datadir):
    nSamples = len(samples)

    if bAugment:
        batchSz //= AUG_MULTIPLIER
    assert batchSz != 0

    while 1:
        shuffle(samples)
        for offset in range(0, nSamples, batchSz):
            batchSamples = samples[offset : offset+batchSz]

            images = []
            angles = []
            for batchSample in batchSamples:
                pathCenter = datadir + '/' + batchSample[0]
                if not os.path.exists(pathCenter):
                    raise('Inconsistent dataset, {} is missing'.format(pathCenter))

                imgCenter = cv2.imread(datadir + '/' + batchSample[0])
                imgLeft = cv2.imread(datadir + '/' + batchSample[1])
                imgRight = cv2.imread(datadir + '/' + batchSample[2])
                angleCenter = float(batchSample[3])
                angleLeft = angleCenter - SIDEVIEW_DELTA_ANGLE
                angleRight = angleCenter + SIDEVIEW_DELTA_ANGLE

                images.append(imgCenter)
                angles.append(angleCenter)

                # augmentations follow
                if bAugment:
                    images.append(np.fliplr(imgCenter))
                    angles.append(-angleCenter)

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
        pathCenter = datadir + '/' + sample[0]
        if not os.path.exists(pathCenter):
            print('Image for entry {} not found, skipping'.format(pathCenter))
            continue
        cleanSamples.append(sample)
    return cleanSamples

def openTelemetry(datadir):
    filename = datadir + '/driving_log.csv'
    if not os.path.exists(filename):
        raise('File could not be opened "{}"'.format(filename))
    samples = pd.read_csv(filename, sep='\s*,\s*', engine='python').values.tolist()
    return reconcileDataset(samples, datadir)

def openTelemetryTrainValidFromSingleDataset(datadir):
    samples = openTelemetry(datadir)
    trainSamples, validSamples = train_test_split(samples, test_size=0.2)
    return trainSamples, validSamples

def openTelemetryTrainValidFromDatasets(dirTrain, dirValid):
    return openSamples(dirTrain), openSamples(dirValid)

def train(dirTrain, dirValid, batchSz, epochs, bResume=False):
    if dirValid is None:
        telemetryTrainSamples, telemetryValidSamples = openTelemetryTrainValidFromSingleDataset(dirTrain)
        dirValid = dirTrain
    else:
        telemetryTrainSamples, telemetryValidSamples = openTelemetryTrainValidFromDatasets(dirTrain, dirValid)

    print('Training with {} training and {} validation samples, batch size {}, epochs {}'.format(len(telemetryTrainSamples), len(telemetryValidSamples), batchSz, epochs))

    trainGenerator = batchGenerator(telemetryTrainSamples, True, batchSz, dirTrain)
    validGenerator = batchGenerator(telemetryValidSamples, False, batchSz, dirValid)

    #TODO: remove these hardcodings, use tee(trainGenerator) and read the first image
    rows, cols, chs = 160, 320, 3

    kerasVer = keras.__version__
    print('Keras version is {}'.format(keras.__version__))

    model = Sequential()
    if StrictVersion(kerasVer) <= StrictVersion('1.2.1'):
        model.add(Cropping2D(cropping=((50,20), (0,0)), dim_ordering='tf', input_shape=(rows, cols, chs)))
    else:
        model.add(Cropping2D(cropping=((50,20), (0,0)), data_format='channels_last', input_shape=(rows, cols, chs)))
    model.add(Lambda(lambda x: x/127.5-1.0))
    model.add(Flatten())
    model.add(Dense(1))

    modelFilename = 'model.h5'
    if os.path.exists(modelFilename) and bResume:
        model.load_weights(modelFilename)
        print('Loaded model from file')

    model.compile(loss='mse', optimizer='adam')

    checkpointer = ModelCheckpoint(filepath=modelFilename, verbose=1, save_best_only=True)
    samplesPerEpoch = len(telemetryTrainSamples) * AUG_MULTIPLIER
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

train(args.train, args.valid, args.batch, args.epochs)

