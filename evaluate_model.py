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
# import pandas as pd
# from nmp.dataset import pyplot_piano_roll, write_midi
# import matplotlib.pyplot as plt

P = Path(os.path.abspath(''))  # Compatible with Jupyter Notebook

PLOTS = P / 'plots'  # Plots path
BS = 32
FS = 24  # Sampling frequency. 10 Hz = 100 ms
Q = 0  # Quantize?
st = 10  # Past timesteps
num_ts = 10  # Predicted timesteps
DOWN = 12  # Downsampling factor
D = "data/piano-midi.de/test"  # Dataset (synth or data)
# MODEL = 'model-LSTM-24-10-12'
# MODEL = 'chorales-ff-2'
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
    test_list = midi_list  # [23:25]
    print(test_list)
    test = dataset.Dataset(test_list, P / D,  fs=FS, bl=0, quant=Q)
    test.build_dataset("test", step=st, t_step=num_ts, steps=st, down=DOWN,
                       low_lim=LOW_LIM, high_lim=HIGH_LIM)

    baseline = dataset.Dataset(test_list, P / D,  fs=FS, bl=1, quant=Q)
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

    # # AUC
    # pred_auc = ev_metrics.compute_auc(test.dataset[1], predictions,
    #                                   NUM_NOTES)
    # print("Mean AUC: %.2f" % np.nanmean(pred_auc))

    # ranked_predictions = dataset.ranked_threshold(predictions, steps=10)
    # score = ev_metrics.dissonance_perception(test.dataset[1],
    #                                          ranked_predictions)
    # print("Perception score: %.2f" % score)

    # base_score = ev_metrics.dissonance_perception(test.dataset[1][:, :64],
    #                                              baseline.dataset[1][:, :64])

    # print("Perception score: %.2f" % score)
    # print("Baseline perception score: %.2f" % base_score)

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
        # print(a.shape)
        # print(b.shape)
    print(np.mean(score))

    # print("Computing best threshold...")
    # # best_thresh = None
    # best_thresh = 0.14
    # if not best_thresh:
    #     (best_thresh,
    #      best_res,
    #      thresh_range,
    #      results) = ev_metrics.compute_best_thresh(test.dataset[1],
    #                                                predictions,
    #                                                NUM_NOTES)
    #     print("Best threshold = %.2f - F1-score: %.4f" % (best_thresh,
    #                                                       best_res))
    #     plt.figure(figsize=(6, 4), constrained_layout=True)
    #     plt.plot(thresh_range, results, label="Predictions", lw=2)
    #     plt.xlabel("Threshold")
    #     plt.ylabel("F1-score")
    #     plt.title("F1-score vs Threshold")
    #     plt.ylim([0, 1])
    #     plt.legend()

    # else:
    #     print("Selected threshold: %.2f" % best_thresh)


if __name__ == '__main__':
    main()
