#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset functions."""
import pandas as pd
import numpy as np
import librosa.display
import pretty_midi as pm


def transpose(data):
    """Transpose time and pitch."""
    df = pd.DataFrame(data)
    return df.T.values


def binarize(data):
    """Binarize data."""
    data = np.array([[1 if e else 0 for e in r] for r in data])
    return data


def plot_piano_roll(pr, start_pitch, end_pitch, ax, fs=100):
    """Plot piano roll representation."""
    librosa.display.specshow(pr[start_pitch:end_pitch], hop_length=1, sr=fs,
                             x_axis='time', y_axis='cqt_note',
                             fmin=pm.note_number_to_hz(start_pitch),
                             ax=ax)
