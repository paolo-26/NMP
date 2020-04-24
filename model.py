#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create model."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
import keras.metrics


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
                           keras.metrics.Recall()])

    return model
