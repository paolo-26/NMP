#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create model."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
import keras.metrics
import keras.backend as K


def build_model():
    """Create Keras model."""
    model = Sequential()
    model.add(Dense(80))
    model.add(Dense(60))
    # model.add(SimpleRNN(64))
    model.add(Dense(128, activation='softmax'))

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
