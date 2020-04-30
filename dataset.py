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


class DataGenerator:
    """Class to yield midi_file as batches."""

    def __init__(self, midi_list, path, fs=4, quant=0):
        """Initialise list."""
        self.midi_list = midi_list
        self.dim = len(midi_list)
        self.fs = fs
        self.quant = quant
        self.path = path

    def quantize(self, obj):
        """Quantize time axis."""
        PPQ = obj.resolution
        for instr in obj.instruments:
            for note in instr.notes:
                note.start = obj.time_to_tick(note.start)/PPQ
                note.end = obj.time_to_tick(note.end)/PPQ

        return obj

    def generate(self, step=1, limit=None):
        """Yield the entire dataset infinite times.

        Target and data differ by one timestep.
        If limit==1: stop generator after 1 epoch.
        """
        while True:
            for m in self.midi_list:
                midi_object = pm.PrettyMIDI(str(self.path / m))

                if self.quant:
                    midi_object = self.quantize(midi_object)

                data = transpose(midi_object.get_piano_roll(self.fs))
                target = data[step:, :]
                data = data[:-1, :]
                data_new = []

                for c in range(len(data)-step+1):
                    data_new.append(np.array(data[c:c+step]))

                if step != 1:
                    data = np.array(data_new)

                yield ((data, target))

            if limit == 1:
                break


class DataGeneratorOld:
    """Alternative generator which yields batches of equal size."""

    def __init__(self):
        """Initialize."""
        self.flag = 0
        self.dimension = 0

    def load_file(self, filename, fs=4):
        """Load a single file and create inputs and targets data."""
        pr = pm.PrettyMIDI(filename).get_piano_roll(fs)
        pr = transpose(pr)
        self.load_data(pr[1:, :], pr[:-1, :])

    def load_data(self, inputs, targets):
        """Load inputs and targets into database."""
        if self.flag == 0:
            self.inputs = inputs
        else:
            self.inputs = np.concatenate((self.inputs, inputs), axis=0)

        if self.flag == 0:
            self.targets = targets
        else:
            self.targets = np.concatenate((self.targets, targets), axis=0)

        self.flag = 1

        self.dimension += len(inputs)

    def generate_data(self, bs, steps):
        """Yield batches of data."""
        step = 0
        for s in range(steps):
            tup = (self.inputs[step*bs:step*bs+bs],
                   self.targets[step*bs:step*bs+bs])
            step += 1
            print(step*bs, step*bs+bs)
            yield tup
