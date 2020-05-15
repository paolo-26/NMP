#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create model."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import SimpleRNN
import tensorflow as tf
import keras.metrics
import keras.backend as K


def build_model(inp_shape, num_ts):
    """Create Keras model."""
    model = Sequential()
    # model.add(Flatten())
    model.add(Dense(90, input_shape=(inp_shape,),  activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(88*num_ts, activation='sigmoid', name='Output'))

    opt = tf.keras.optimizers.SGD(learning_rate=0.3)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', # opt,
                  metrics=[keras.metrics.Precision(),
                           keras.metrics.Recall(),
                           f1])

    return model


def f1(y_true, y_pred):
    """Compute F1-score metric.

    Taken from old keras source code.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
