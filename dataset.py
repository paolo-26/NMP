#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset functions."""
import pandas as pd
import numpy as np
# import librosa.display
# import pretty_midi as pm
import copy
import pypianoroll


def transpose(data):
    """Transpose time and pitch."""
    df = pd.DataFrame(data)
    return df.T.values


def binarize(data):
    """Binarize data."""
    data = np.array([[1 if e else 0 for e in r] for r in data])
    return data


def convert(data):
    """Convert data from sigmoid output to 0-1."""
    data = np.array([[1 if e > 0.5 else 0 for e in r] for r in data])
    return data


# def plot_piano_roll(pr, start_pitch, end_pitch, ax, fs=100):
#     """Plot piano roll representation."""
#     librosa.display.specshow(pr[start_pitch:end_pitch], hop_length=1, sr=fs,
#                              x_axis='time', y_axis='cqt_note',
#                              fmin=pm.note_number_to_hz(start_pitch),
#                              ax=ax)

def pad_piano_roll(pr):
    """Pad cropped piano roll."""
    L = len(pr)
    pad1 = np.zeros((L, 21))
    pad2 = np.zeros((L, 19))
    pr = np.concatenate((pad1, pr, pad2), axis=1)
    return pr


def pyplot_piano_roll(pr, cmap="Blues"):
    """Plot piano roll representation."""
    pr = pad_piano_roll(pr)
    pr = pypianoroll.Track(pianoroll=pr)
    return pypianoroll.plot_track(pr, cmap=cmap)


def import_one(filename, beat_resolution):
    """Import one file and generate a piano roll."""
    pr = pypianoroll.parse(filename, beat_resolution)
    merged = pr.get_merged_pianoroll()
    return merged


class DataGenerator:
    """Class to yield midi_file as batches."""

    def __init__(self, midi_list, path, fs=4, quant=0, binarize=1, bl=0):
        """Initialise list."""
        self.midi_list = midi_list
        self.dim = len(midi_list)
        self.fs = fs
        self.quant = quant
        self.binar = binarize
        self.path = path
        self.baseline = bl

    def binarize(self, pr):
        """Binarize data."""
        pr = np.array([[np.float32(1) if e else np.float32(0) for e in r]
                       for r in pr])
        return pr

    def quantize(self, obj):
        """Quantize time axis."""
        PPQ = obj.resolution
        for instr in obj.instruments:
            for note in instr.notes:
                note.start = obj.time_to_tick(note.start)/PPQ
                note.end = obj.time_to_tick(note.end)/PPQ

        return obj

    def pitch_transpose(self, ds, value, timesteps=None):
        """Perform pitch transposition."""
        if timesteps:
            range_timesteps = ds.shape[1]
            for t in range(range_timesteps):
                d = pd.DataFrame(ds[:, t, :])
                d.shift(value, axis=1, fill_value=0)
                ds[:, t, :] = d.values
        else:
            d = pd.DataFrame(ds)
            d.shift(value, axis=1, fill_value=0)
            ds = d.values

        return ds

    def generate(self, bs, limit=None, trans=None, name=''):
        """Yield a dataset."""
        dim = self.dataset[0].shape[0]
        batch = bs
        epoch = 0
        while True:
            cnt = 0
            epoch += 1
            en = 0
            c = 0
            while en < dim:
                en = c + batch
                yield (self.dataset[0][c:en], self.dataset[1][c:en])
                c += batch
                cnt += 1
            if trans == 1:
                for v in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    x = self.pitch_transpose(self.dataset[0], v, 1)
                    y = self.pitch_transpose(self.dataset[1], v)
                    en = 0
                    c = 0
                    while en < dim:
                        en = c + batch
                        yield (x[c:en], y[c:en])
                        c += batch
                        cnt += 1
            print("\nEpoch %d finished for %s, %d" % (epoch, name, cnt))
            # if limit == 1:
            #     break

    # def import_one_pretty(self, filename, fs, quant=0, binar=0):
    #     """Import one file and generate a piano roll."""
    #     midi_object = pm.PrettyMIDI(filename)
    #
    #     if quant:
    #         midi_object = self.quantize(midi_object)
    #
    #     pr = midi_object.get_piano_roll(fs)
    #     prt = transpose(pr[21:109, :])
    #
    #     if binar:
    #         prt = self.binarize(prt)
    #
    #     return prt

    def generate_transposed(self):
        """Create a generator for transposed dataset."""
        # for v in [-5, -3, -1, 1, 3, 5]:
        for v in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
            x = self.pitch_transpose(self.dataset[0], v, 1)
            y = self.pitch_transpose(self.dataset[1], v)
            yield (x, y)

    def build_transposed(self):
        """Build a transposed dataset."""
        generator = self.generate_transposed()
        # print(self.dataset[0].shape)
        x_data = self.dataset[0]
        y_target = self.dataset[1]

        for i in range(6):
            (x, y) = next(generator)

            # Append transposed data.
            x_data = np.concatenate((x_data, x), axis=0)
            # Append transposed target.
            y_target = np.concatenate((y_target, y), axis=0)

        self.dataset = (x_data, y_target)

    def build_dataset(self, name, step=1, t_step=1):
        """Build a dataset."""
        print("Building %s dataset (%d files)" % (name, len(self.midi_list)))
        flag = 0
        for m in self.midi_list:

            prt = import_one(str(self.path / m), beat_resolution=self.fs)
            prt = prt[:, 21:109]
            prt = self.binarize(prt)

            # prt = self.import_one_pretty(str(self.path / m),
            #                              self.fs, self.quant, self.binar)
            if self.baseline:
                data = prt[:-t_step, :]
                target = prt[step-1:-1, :]

            else:
                data = prt[:-t_step, :]
                target = prt[step:, :]

            if step > 0:
                data_new = []
                for c in range(len(data)-step+1):
                    conc = np.array([data[x] for x in range(c, c+step)])
                    data_new.append(conc)

                data = np.array(data_new)

            if t_step > 0:
                target_new = []

                if self.baseline:
                    for c in range(len(target)-t_step+1):
                        conc = np.array([target[c] for x in range(c,
                                                                  c+t_step)])

                        target_new.append(np.concatenate(conc, axis=None))

                else:
                    for c in range(len(target)-t_step+1):
                        conc = np.array([target[x] for x in range(c,
                                                                  c+t_step)])
                        target_new.append(np.concatenate(conc, axis=None))

                target = np.array(target_new)

            if flag == 0:
                a = copy.copy(data)
                b = copy.copy(target)
                flag = 1

            else:
                a2 = copy.copy(a)
                b2 = copy.copy(b)
                a = np.concatenate((a2, data), axis=0)
                b = np.concatenate((b2, target), axis=0)

                data = copy.copy(a)
                target = copy.copy(b)

        self.dataset = ((data, target))
        self.dime = len(data)
