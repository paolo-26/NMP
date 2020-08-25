#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Display some predictions."""
import os
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
from nmp import dataset, ev_metrics
# from tensorflow.keras.layers import LSTM
import tensorflow as tf
import copy
import pandas as pd
from nmp.dataset import pyplot_piano_roll
import matplotlib.pyplot as plt

P = Path(os.path.abspath(''))  # Compatible with Jupyter Notebook

PLOTS = P / 'plots'  # Plots path
BS = 64
FS = 24  # Sampling frequency. 10 Hz = 100 ms
Q = 0  # Quantize?
st = 10  # Past timesteps
num_ts = 10  # Predicted timesteps
DOWN = 12  # Downsampling factor
D = "data"  # Dataset (synth or data)
# MODEL = 'model-LSTM-24-10-12'
MODEL = 'temp-reduced'
LOW_LIM = 38
HIGH_LIM = 85
NUM_NOTES = HIGH_LIM - LOW_LIM
CROP = [LOW_LIM, HIGH_LIM]  # Crop plots

# TensorFlow stuff
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    # Load midi files
    midi_list = [x for x in os.listdir(P / D) if x.endswith('.mid')]
    test_list = midi_list[213:235]
    test = dataset.Dataset(test_list, P / D,  fs=FS, bl=0, quant=Q)
    test.build_dataset("test", step=st, t_step=num_ts, steps=st, down=DOWN,
                       low_lim=LOW_LIM, high_lim=HIGH_LIM)

    # Load model.
    model = load_model(filepath=str(P / 'models' / MODEL),
                       custom_objects=None, compile=True)
    model.summary()

    # Make predictions.
    predictions = model.predict(x=test.dataset[0])
    pred_auc = ev_metrics.compute_auc(test.dataset[1], predictions, NUM_NOTES)
    print(np.nanmean(pred_auc))

    best_thresh, best_res = ev_metrics.compute_best_thresh(test.dataset[1],
                                                           predictions,
                                                           NUM_NOTES)
    print("Best threshold = %.1f - F1-score: %.4f" % (best_thresh, best_res))

    # Select one file
    FILE = 'bach_846cut2.mid'
    test_list = [P / 'midi_tests' / FILE]

    test = dataset.Dataset(test_list, P / D,  fs=FS, bl=0, quant=Q)
    test.build_dataset("test", step=st, t_step=num_ts, steps=st, down=DOWN,
                       low_lim=LOW_LIM, high_lim=HIGH_LIM)
    predictions = model.predict(x=test.dataset[0])
    predictions_bin = dataset.threshold(predictions, best_thresh)

    # Concatenate piano rolls.
    newpiano = copy.deepcopy(predictions_bin)
    L = int(predictions_bin.shape[1]/NUM_NOTES)
    for t in range(L):
        newpiano[newpiano.shape[0]-L+t,
                 :NUM_NOTES] = predictions_bin[newpiano.shape[0]-L,
                                               NUM_NOTES*t:NUM_NOTES*(t+1)]

    a = copy.deepcopy(test.dataset[1][:, :])
    L = int(test.dataset[1][:, :].shape[1]/NUM_NOTES)
    for t in range(L):
        a[a.shape[0]-L+t,
          :NUM_NOTES] = test.dataset[1][a.shape[0]-L,
                                        NUM_NOTES*t:NUM_NOTES*(t+1)]

    a = pd.DataFrame(test.dataset[0][:, 0, :])
    b = pd.DataFrame(newpiano[-num_ts:, :NUM_NOTES])
    b2 = pd.DataFrame(test.dataset[1][-num_ts:, :NUM_NOTES])

    c = pd.concat([a, b]).values
    d = pd.concat([a, b2]).values

    plt.rcParams["figure.figsize"] = (8, 4)
    pyplot_piano_roll(d, cmap="Greens", db=[d.shape[0]-L-0.5], br=2,
                      low_lim=LOW_LIM, high_lim=HIGH_LIM)
    plt.title('Real')
    plt.ylim(CROP)
    # plt.xlim([50, 71])

    pyplot_piano_roll(c, cmap="Oranges", db=[d.shape[0]-L-0.5], br=2,
                      low_lim=LOW_LIM, high_lim=HIGH_LIM)
    plt.title('Prediction (thresholded)')
    plt.ylim(CROP)
    # plt.xlim([50, 71])
    plt.show()


if __name__ == '__main__':
    main()
