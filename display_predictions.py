#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Display some predictions."""
import os
from pathlib import Path
# import numpy as np
from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
from nmp import dataset
# from tensorflow.keras.layers import LSTM
import tensorflow as tf
import copy
import pandas as pd
from nmp.dataset import pyplot_piano_roll, write_midi
import matplotlib.pyplot as plt

P = Path(os.path.abspath(''))  # Compatible with Jupyter Notebook

PLOTS = P / 'plots'  # Plots path
BS = 64
FS = 24  # Sampling frequency. 10 Hz = 100 ms
Q = 0  # Quantize?
st = 10  # Past timesteps
num_ts = 10  # Predicted timesteps
DOWN = 12  # Downsampling factor
D = "data/midi_tests"  # Dataset (synth or data)
# MODEL = 'model-LSTM-24-10-12'
# MODEL = 'chorales-ff-2'
MODEL = 'ff-z2-de'

LOW_LIM = 33  # A1
HIGH_LIM = 97  # C7

NUM_NOTES = HIGH_LIM - LOW_LIM
CROP = [LOW_LIM, HIGH_LIM]  # Crop plots

STOP = 18  # Timestep at which interruption occurs

# TensorFlow stuff
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    # Load model.
    model = load_model(filepath=str(P / 'models' / MODEL),
                       custom_objects=None, compile=True)
    model.summary()

    tempo = 130

    # midi-test
    FILE = 'bach-minuetto2.mid'
    test_list = [P / 'midi_tests' / FILE]
    test = dataset.Dataset(test_list, P / D,  fs=FS, bl=0, quant=Q)
    test.build_dataset("test", step=st, t_step=num_ts, steps=st, down=DOWN,
                       low_lim=LOW_LIM, high_lim=HIGH_LIM)

    # Truncate dataset.
    # L = test.dataset[0].shape[0] - (test.dataset[0].shape[0] % BS)
    # x = test.dataset[0][:L, :, :]
    # y = test.dataset[1][:L, :]
    # test.dataset = (x, y)

    # Chorales
    # FILE = 'jsb-chorales-quarter.pkl'
    # test_list = [P / 'data/JSB-Chorales-dataset' / FILE]
    # test = dataset.Dataset(test_list, P / D,  fs=FS, bl=0, quant=Q)
    # test.build_choral("test", step=st, t_step=num_ts, steps=st,
    #                   low_lim=LOW_LIM, high_lim=HIGH_LIM)

    # Predictions
    # print(test.dataset[0][:15, :, :].shape)
    predictions = model.predict(x=test.dataset[0])  # , batch_size=BS)
    # predictions_bin = dataset.threshold(predictions, best_thresh)
    predictions_bin = dataset.ranked_threshold(predictions, steps=10,
                                               how_many=3)

    # Concatenate piano rolls.

    # Sliding window predictions.
    pred_w = copy.deepcopy(predictions_bin)

    # Snapshot predictions.
    pred_snap = copy.deepcopy(predictions_bin)
    L = int(predictions_bin.shape[1]/NUM_NOTES)
    for t in range(L):
        pred_snap[STOP-num_ts+t,
                  :NUM_NOTES] = predictions_bin[STOP-num_ts,
                                                NUM_NOTES*t:NUM_NOTES*(t+1)]

    # Real piano roll of the song until interruption.
    real_start = pd.DataFrame(test.dataset[0][:STOP, 0, :])

    # Real piano roll of the song from interruption.
    real_end = pd.DataFrame(test.dataset[1][STOP-num_ts:STOP, :NUM_NOTES])

    # Predictions piano roll from interruption.
    pred_end = pd.DataFrame(pred_snap[-num_ts:, :NUM_NOTES])  # Snapshot
    pred_end_w = pd.DataFrame(pred_w[-num_ts:, :NUM_NOTES])  # Sliding window

    # Real piano roll before interruption + piano roll after interruption.
    real = pd.concat([real_start, real_end]).values

    # Real piano roll before interruption + predictions after interruption.
    predicted = pd.concat([real_start, pred_end]).values  # Snapshot
    predicted_w = pd.concat([real_start, pred_end_w]).values  # Sliding window

    # Random baseline
    random_baseline = dataset.random_baseline(10, NUM_NOTES)
    random_baseline = pd.concat([real_start, random_baseline]).values

    # Semi-random baseline
    _, noteset = dataset.get_indexes(real_start)
    sr_baseline = dataset.random_baseline(10, NUM_NOTES, noteset)
    sr_baseline = pd.concat([real_start, sr_baseline]).values

    # Hold baseline
    _, noteset = dataset.get_indexes(real_start.tail(1))
    hold_baseline = dataset.hold_baseline(10, NUM_NOTES, noteset)
    hold_baseline = pd.concat([real_start, hold_baseline]).values

    plt.rcParams["figure.figsize"] = (8, 4)
    pyplot_piano_roll(real, cmap="Greens", db=[real.shape[0]-L-0.5], br=2,
                      low_lim=LOW_LIM, high_lim=HIGH_LIM)
    plt.title('Real')
    plt.ylim(CROP)
    plt.xlim([STOP-10, STOP+10])

    pyplot_piano_roll(predicted, cmap="Oranges", db=[real.shape[0]-L-0.5],
                      br=2, low_lim=LOW_LIM, high_lim=HIGH_LIM)
    plt.title('Prediction (snapshot)')
    plt.ylim(CROP)
    plt.xlim([STOP-10, STOP+10])

    pyplot_piano_roll(predicted_w, cmap="Purples", db=[real.shape[0]-L-0.5],
                      br=2, low_lim=LOW_LIM, high_lim=HIGH_LIM)
    plt.title('Prediction (sliding window)')
    plt.ylim(CROP)
    plt.xlim([STOP-10, STOP+10])

    pyplot_piano_roll(random_baseline, cmap="Purples",
                      db=[real.shape[0]-L-0.5],
                      br=2, low_lim=LOW_LIM, high_lim=HIGH_LIM)
    plt.title('Random baseline')
    plt.ylim(CROP)
    plt.xlim([STOP-10, STOP+10])

    pyplot_piano_roll(sr_baseline, cmap="Purples", db=[real.shape[0]-L-0.5],
                      br=2, low_lim=LOW_LIM, high_lim=HIGH_LIM)
    plt.title('Semi-random baseline')
    plt.ylim(CROP)
    plt.xlim([STOP-10, STOP+10])

    # Save piano roll
    # f0 = copy.deepcopy(real)
    # write_midi(f0, str(P / 'audio_output' / 'real.mid'),
    #            LOW_LIM, HIGH_LIM, tempo=tempo)
    f1 = copy.deepcopy(predicted)
    write_midi(f1, str(P / 'audio_output' / 'snapshot-window.mid'),
               LOW_LIM, HIGH_LIM, tempo=tempo)
    f2 = copy.deepcopy(predicted_w)
    write_midi(f2, str(P / 'audio_output' / 'sliding-window.mid'),
               LOW_LIM, HIGH_LIM, tempo=tempo)
    f3 = copy.deepcopy(random_baseline)
    write_midi(f3, str(P / 'audio_output' / 'random.mid'),
               LOW_LIM, HIGH_LIM, tempo=tempo)
    f4 = copy.deepcopy(sr_baseline)
    write_midi(f4, str(P / 'audio_output' / 'semi-random.mid'),
               LOW_LIM, HIGH_LIM, tempo=tempo)
    f5 = copy.deepcopy(hold_baseline)
    write_midi(f5, str(P / 'audio_output' / 'hold.mid'),
               LOW_LIM, HIGH_LIM, tempo=tempo)
    plt.show()


if __name__ == '__main__':
    main()
