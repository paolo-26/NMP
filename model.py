#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create model."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import SimpleRNN
import tensorflow as tf
import keras.metrics
import keras.backend as K


def build_model(inp_shape, num_ts):
    num_ts = int(num_ts)
    """Create Keras model."""
    # Allow memory growth.
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Create the model.
    model = Sequential()
    model.add(Dense(32, input_shape=(inp_shape),  activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(88*num_ts, activation='sigmoid', name='Output'))

    # opt = tf.keras.optimizers.SGD(learning_rate=0.1)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[keras.metrics.Precision(),
                           keras.metrics.Recall(),
                           f1_first, f1_last])

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


def f1_last(y_true, y_pred):
    """Compute F1-score metric on a portion of the output."""
    # Select the last 88 notes.
    y_true = y_true[:, -88:]
    y_pred = y_pred[:, -88:]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def f1_first(y_true, y_pred):
    """Compute F1-score metric on a portion of the output."""
    # Select the first 88 notes.
    y_true = y_true[:, 0:88]
    y_pred = y_pred[:, 0:88]

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def f1_2(y_true, y_pred):
    """Compute F1-score metric on a portion of the output."""
    # Select the second 88 notes.
    y_true = y_true[:, 88:176]
    y_pred = y_pred[:, 88:176]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def f1_3(y_true, y_pred):
    """Compute F1-score metric on a portion of the output."""
    # Select the second 88 notes.
    y_true = y_true[:, 88*2:88*3]
    y_pred = y_pred[:, 88*2:88*3]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def f1_4(y_true, y_pred):
    """Compute F1-score metric on a portion of the output."""
    # Select the second 88 notes.
    y_true = y_true[:, 88*3:88*4]
    y_pred = y_pred[:, 88*3:88*4]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def f1_5(y_true, y_pred):
    """Compute F1-score metric on a portion of the output."""
    # Select the second 88 notes.
    y_true = y_true[:, 88*4:88*5]
    y_pred = y_pred[:, 88*4:88*5]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def f1_6(y_true, y_pred):
    """Compute F1-score metric on a portion of the output."""
    # Select the second 88 notes.
    y_true = y_true[:, 88*5:88*6]
    y_pred = y_pred[:, 88*5:88*6]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def f1_7(y_true, y_pred):
    """Compute F1-score metric on a portion of the output."""
    # Select the second 88 notes.
    y_true = y_true[:, 88*6:88*7]
    y_pred = y_pred[:, 88*6:88*7]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def f1_8(y_true, y_pred):
    """Compute F1-score metric on a portion of the output."""
    # Select the second 88 notes.
    y_true = y_true[:, 88*7:88*8]
    y_pred = y_pred[:, 88*7:88*8]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def f1_9(y_true, y_pred):
    """Compute F1-score metric on a portion of the output."""
    # Select the second 88 notes.
    y_true = y_true[:, 88*8:88*9]
    y_pred = y_pred[:, 88*8:88*9]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
