#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create model."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import SimpleRNN
import tensorflow as tf
# import keras.metrics

# Allow memory growth.
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def build_model(inp_shape, num_ts):
    num_ts = int(num_ts)
    """Create Keras model."""

    # Create the model.
    model = Sequential()
    model.add(Dense(32, input_shape=(inp_shape),  activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(88*num_ts, activation='sigmoid', name='Output'))

    # Create the model with RNN.
    # model = Sequential()
    # model.add(SimpleRNN(32, return_sequences=True,
    #                     input_shape=(inp_shape),  activation='relu'))
    # model.add(SimpleRNN(32, return_sequences=True, activation='relu'))
    # model.add(SimpleRNN(88*num_ts, activation='sigmoid', name='Output'))

    # opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    return model


def compile_model(model, loss, optimizer, metrics):
    """Compile Keras model."""
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
