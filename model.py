#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create model."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from keras.models import model_from_yaml
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import SimpleRNN
import tensorflow as tf
import keras.metrics
import keras.backend as K
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


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

    # Create the model with RNN.
    # model = Sequential()
    # model.add(SimpleRNN(32, return_sequences=True,
    #                     input_shape=(inp_shape),  activation='relu'))
    # model.add(SimpleRNN(32, return_sequences=True, activation='relu'))
    # model.add(SimpleRNN(88*num_ts, activation='sigmoid', name='Output'))

    # opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    return model


def compile_model(model, loss, optimizer):
    """Compile Keras model."""
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[keras.metrics.Precision(),
                           keras.metrics.Recall(),
                           f1_first, f1_last])


def save_model(model, path):
    """Save model to disk."""
    # Serialize model to YAML.
    model_yaml = model.to_yaml()
    with open(path / 'model.yaml', 'w') as yaml_file:
        yaml_file.write(model_yaml)

    # Serialize weights to HDF5.
    model.save_weights(str(path / 'model.h5'))
    print("Saved model to %s." % path)


def load_model(path):
    """Load model from file."""
    # Load YAML and create model.
    print(path / 'model.yaml')
    with open(path / 'model.yaml', 'r') as yaml_file:
        loaded_model_yaml = yaml_file.read()

    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        loaded_model = model_from_yaml(loaded_model_yaml)

    # Load weights into new models.
    loaded_model.load_weights(str(path / 'model.h5'))
    print("Loaded model from disk.")

    return loaded_model


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
