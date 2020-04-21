#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create model."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_model():
    """Create Keras model."""
    model = Sequential()
    model.add(Dense(80))
    model.add(Dense(64))
    model.add(Dense(128, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
