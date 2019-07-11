from __future__ import absolute_import
from __future__ import print_function
import warnings
import os
import numpy as np
import scipy.io as sio
from subprocess import call
from keras.datasets import mnist, cifar10
from keras.utils import np_utils


def get_data(dataset='mnist', n_class=10, clip_min=0.0, clip_max=1.0, onehot=True, path='data/'):
    """
    Load datasets (automatically download if not exist) and normalize to the range of [clip_min, clip_max].
    """
    if dataset == 'mnist':
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # reshape to (n_samples, 28, 28, 1)
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
    elif dataset == 'cifar-10':
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    else:
        print("Add new type of dataset here such as cifar-100.")
        return

    # cast pixels to floats, normalize to [0, 1] range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = (X_train / 255.0) - (1.0 - clip_max)
    X_test = (X_test / 255.0) - (1.0 - clip_max)

    # of only load a few classes
    if n_class > 0 and n_class < np.max(y_train) + 1:
        train_sample_idx = np.where(y_train < n_class)[0]
        X_train = X_train[train_sample_idx]
        y_train = y_train[train_sample_idx]
        test_sample_idx = np.where(y_test < n_class)[0]
        X_test = X_test[test_sample_idx]
        y_test = y_test[test_sample_idx]
    else:
        n_class = np.max(y_train) + 1

    # one-hot-encode the labels
    if onehot:
        Y_train = np_utils.to_categorical(y_train, n_class)
        Y_test = np_utils.to_categorical(y_test, n_class)
    else:
        Y_train = y_train
        Y_test = y_test

    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_test:", X_test.shape)
    print("Y_test", Y_test.shape)

    return X_train, Y_train, X_test, Y_test
