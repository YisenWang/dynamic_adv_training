# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from keras import backend as K
import tensorflow as tf

def cross_entropy(y_true, y_pred):
    y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
