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
    # Allow memory growth.
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Create the model.
    model = Sequential()
    model.add(Dense(64, input_shape=(inp_shape,),  activation='relu'))
    model.add(Dense(64, input_shape=(inp_shape,),  activation='relu'))
    model.add(Dense(88, activation='sigmoid', name='Output'))

    # opt = tf.keras.optimizers.SGD(learning_rate=0.1)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
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
