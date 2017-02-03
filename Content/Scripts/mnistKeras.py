'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
3 seconds per epoch on a Titan X Pascal.

ue.exec('mnistKeras.py')
'''

from __future__ import print_function
import numpy as np

class Logwrapper(object):
    def __init__(self):
        self.terminal = ue.log

    def write(self, message):
        ue.log(message)

    def flush(self):
        ue.log("")

import unreal_engine as ue
import sys

import _thread as thread

#wrap default logs so we get all print()
sys.stdout = Logwrapper()
sys.stderr = Logwrapper()

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

def train():
    print("Training started...")

    np.random.seed(1337)  # for reproducibility
    batch_size = 128
    nb_classes = 10
    nb_epoch = 12

    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    ue.log('X_train shape:' + str(X_train.shape))
    ue.log(str(X_train.shape[0]) + 'train samples')
    ue.log(str(X_test.shape[0]) + 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    print(X_train)

    print(Y_train)

    #model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #          verbose=1, validation_data=(X_test, Y_test))
    #score = model.evaluate(X_test, Y_test, verbose=0)
    #ue.log('Test score:' + str(score[0]))
    #ue.log('Test accuracy:' + str(score[1]))

#start thread
thread.start_new_thread(train, ())

#train()