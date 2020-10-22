#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create model."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import keras.backend as K

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
    # model.add(SimpleRNN(units=32,
    #                batch_input_shape=(bs, 10, 64),
    #                return_sequences=True,
    #                stateful=True,
    #                input_shape=(inp_shape),
    #                activation='relu'))
    # model.add(SimpleRNN(32, activation='relu'))
    # model.add(Dense(num_notes*num_ts, activation='sigmoid', name='Output'))

    # LSTM.
    # model = Sequential()
    # model.add(LSTM(32,  input_shape=(inp_shape), return_sequences=True,
    #                activation='relu'))
    # model.add(LSTM(32, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(num_notes*num_ts, activation='sigmoid', name='Output'))

    return model


def build_gru_model(num_notes, batch_size):
    """Create Keras model."""
    model = Sequential(
        [Input(batch_input_shape=[batch_size, None, num_notes]),

         LSTM(units=32,
             return_sequences=True,
             stateful=True),
         LSTM(units=16,
             return_sequences=True,
             stateful=True),
         Dense(units=num_notes,
               activation='sigmoid',
               name='Output')
         ]
    )

    return model


def compile_model(model, loss, optimizer, metrics):
    """Compile Keras model."""
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)


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
