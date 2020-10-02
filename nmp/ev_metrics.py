#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation metrics."""
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from math import nan
from nmp.dataset import threshold, get_indexes
import numpy as np


def compute_auc(x, y, num_notes):
    """Compute AUC-ROC for every timestep and return the list of AUCs.

    Averaged over pitches.

    x: true values
    y: score values

    Returns a dataframe of shape (88, n_ft)
    """
    n_ft = int(x.shape[1]/num_notes)  # Number of future timesteps
    start = 0
    end = start + num_notes
    auc = []
    aucs = []

    for _ in range(n_ft):
        auc = []
        for p in range(num_notes-1, -1, -1):

            try:
                auc.append(roc_auc_score(x[:, start+p],  y[:, start+p]))

            except Exception:
                # There are no played notes.
                auc.append(nan)

        aucs.append(auc)
        start += num_notes  # Next predicted timestep
        end += num_notes  # Next predicted timestep

    auc_df = pd.DataFrame(aucs).transpose()
    auc_df.index = list(range(num_notes, 0, -1))
    return auc_df


def compute_f1(x, y, num_notes):
    """Compute F1-score for every timestep and return the list of F1 scores.

    Averaged over pitches.

    x: true values
    y: score values

    Returns a dataframe of shape (88, n_ft)
    """
    n_ft = int(x.shape[1]/num_notes)  # Number of future timesteps
    start = 0
    end = start + num_notes
    f1 = []
    f1s = []

    for _ in range(n_ft):
        f1 = []
        for p in range(num_notes-1, -1, -1):

            try:
                f1.append(f1_score(x[:, start+p],  y[:, start+p]))

            except Exception:
                # There are no played notes.
                f1.append(nan)

        f1s.append(f1)
        start += num_notes  # Next predicted timestep
        end += num_notes  # Next predicted timestep

    f1_df = pd.DataFrame(f1s).transpose()
    f1_df.index = list(range(num_notes, 0, -1))
    return f1_df


def compute_best_thresh(x, y, num_notes):
    """Compute best threshold."""

    best_thresh = None
    best_res = 0
    results = []
    thresh_range = np.arange(0.02, 0.3, 0.01)
    for thresh in thresh_range:
        pred_f1 = compute_f1(x, threshold(y, thresh), num_notes)
        res = np.nanmean(pred_f1)
        results.append(res)
        print("%.2f - F1: %.4f" % (thresh, res))
        if res > best_res:
            best_thresh = thresh
            best_res = res

    return (best_thresh, best_res, thresh_range, results)


def dissonance_perception(x, y):
    """Compute dissonance_perception metric.

    It still does not work very well.
    """
    scores = {
        0: 0.075,  # Unison
        12: 0.023,  # Octave
        7: 0.022,  # Fifth
        5: 0.012,  # Fourth
        9: 0.010,  # Major sixth
        4: 0.010,  # Major third
        3: 0.010,  # Minor third
        8: 0.007,  # Minor sixth
        2: 0.006,  # Major second
        11: 0.005,  # Major seventh
        10: 0.003,  # Minor seventh
        1: 0,  # Minor second
        6: 0,  # Tritone
    }
    x, _ = get_indexes(x)
    y, _ = get_indexes(y)

    # values = [b[0]-a[0] for a, b in zip(x, y)]
    values = []
    for t, a in enumerate(x):
        # print("Timestep %d" % t)
        v = []
        for real in a:
            # print("Nota %s" % real)
            lista = [b-real for b in y[t]]
            # print("Distanze: ", lista)
            try:
                v.append(min(lista, key=abs))
            except Exception:
                pass  # Silence

        values.append(v)

    finished = 0
    while finished == 0:
        finished = 1
        for c, val in enumerate(values):
            for i, v in enumerate(val):
                if v < 0:
                    values[c][i] += 12
                    finished = 0

                if v > 12:
                    values[c][i] -= 12
                    finished = 0

    score = [scores[v] for val in values for v in val]
    score = np.sum(score) / len(score) / 0.075

    return score
