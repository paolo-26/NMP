#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate consonance perception."""
import os
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from nmp import dataset, ev_metrics
import tensorflow as tf
import copy

P = Path(os.path.abspath(''))  # Compatible with Jupyter Notebook

PLOTS = P / 'plots'  # Plots path
BS = 64
FS = 24  # Sampling frequency. 10 Hz = 100 ms
st = 10  # Past timesteps
num_ts = 10  # Predicted timesteps
DOWN = 12  # Downsampling factor
D = "data/piano-midi.de/test"  # Dataset (synth or data)
MODEL = 'ff-z-de'

LOW_LIM = 33  # A1
HIGH_LIM = 97  # C7

NUM_NOTES = HIGH_LIM - LOW_LIM
CROP = [LOW_LIM, HIGH_LIM]  # Crop plots

STOP = 12  # Timestep at which interruption occurs

# TensorFlow stuff
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    # Load midi files
    midi_list = [x for x in os.listdir(P / D) if x.endswith('.mid')]
    test_list = midi_list
    print(test_list)
    test = dataset.Dataset(test_list, P / D,  fs=FS, bl=0, quant=0)
    test.build_dataset("test", step=st, t_step=num_ts, steps=st, down=DOWN,
                       low_lim=LOW_LIM, high_lim=HIGH_LIM)

    baseline = dataset.Dataset(test_list, P / D,  fs=FS, bl=1, quant=0)
    baseline.build_dataset("test", step=st, t_step=num_ts, steps=st, down=DOWN,
                           low_lim=LOW_LIM, high_lim=HIGH_LIM)

    print(test.dataset[1][:, :64].shape)
    print(baseline.dataset[1][:, :64].shape)

    # Load model.
    model = load_model(filepath=str(P / 'models' / MODEL),
                       custom_objects=None, compile=True)
    model.summary()

    # Make predictions.
    predictions = model.predict(x=test.dataset[0])

    predictions_bin = dataset.ranked_threshold(predictions, steps=10,
                                               how_many=3)

    x = copy.deepcopy(test.dataset[1])
    y = copy.deepcopy(predictions_bin)

    score = []

    for timestep in range(len(x)):
        # Snapshot predictions.
        pred_snap = copy.deepcopy(y)
        L = int(y.shape[1]/NUM_NOTES)
        for t in range(L):
            pred_snap[timestep-num_ts+t,
                      :NUM_NOTES] = y[timestep+10-num_ts,
                                      NUM_NOTES*t:NUM_NOTES*(t+1)]
        a = x[timestep:timestep+10, :64]
        b = pred_snap[timestep:timestep+10, :64]
        score.append(ev_metrics.dissonance_perception(a, b))
        # print(timestep, score[-1])

    print("Dissonance score: ", np.mean(score))


if __name__ == '__main__':
    main()
