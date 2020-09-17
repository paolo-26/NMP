#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create model."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import SimpleRNN
# from tensorflow.keras.layers import LSTM
import tensorflow as tf

# Allow memory growth.
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def build_model(inp_shape, num_ts, num_notes, bs):
    """Create Keras model."""

    num_ts = int(num_ts)
    # opt = tf.keras.optimizers.SGD(learning_rate=0.1)

    # Feedforward.
    model = Sequential()
    model.add(Dense(32, input_shape=(inp_shape),  activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_notes*num_ts, activation='sigmoid', name='Output'))

    # RNN.
    # model = Sequential()
    # model.add(SimpleRNN(units=16,
    #                     batch_input_shape=(bs, 10, 64),
    #                     return_sequences=True,
    #                     stateful=True,
    #                     input_shape=(inp_shape),
    #                     activation='relu'))
    # model.add(SimpleRNN(16, activation='relu'))
    # model.add(Dense(num_notes*num_ts, activation='sigmoid', name='Output'))

    # LSTM.
    # model = Sequential()
    # model.add(LSTM(16,  input_shape=(inp_shape), return_sequences=True,
    #                activation='relu'))
    # model.add(LSTM(16, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_notes*num_ts, activation='sigmoid', name='Output'))

    return model


def compile_model(model, loss, optimizer, metrics):
    """Compile Keras model."""
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
