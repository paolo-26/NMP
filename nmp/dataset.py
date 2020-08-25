#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset functions."""
import pandas as pd
import numpy as np
import pypianoroll
import random


def transpose(data):
    """Transpose time and pitch."""
    df = pd.DataFrame(data)
    return df.T.values


def binarize(data):
    """Binarize data."""
    return np.array([[1 if e else 0 for e in r] for r in data])


def threshold(data, thresh=0.5):
    """Convert data from sigmoid output to 0-1."""
    return np.array([[1 if e > thresh else 0 for e in r] for r in data])


# def plot_piano_roll(pr, start_pitch, end_pitch, ax, fs=100):
#     """Plot piano roll representation."""
#     librosa.display.specshow(pr[start_pitch:end_pitch], hop_length=1, sr=fs,
#                              x_axis='time', y_axis='cqt_note',
#                              fmin=pm.note_number_to_hz(start_pitch),
#                              ax=ax)

def pad_piano_roll(pr, low_lim=21, high_lim=109):
    """Convert 88-notes piano roll to 128-notes piano roll.

    Useful for plotting using pypianoroll library.
    """
    L = len(pr)
    pad1 = np.zeros((L, low_lim))
    pad2 = np.zeros((L, 128 - high_lim))
    pr = np.concatenate((pad1, pr, pad2), axis=1)
    return pr


def pyplot_piano_roll(pr, cmap="Blues", br=None, db=None, low_lim=21,
                      high_lim=109):
    """Plot piano roll representation."""
    pr = pad_piano_roll(pr, low_lim, high_lim)
    pr = pypianoroll.Track(pianoroll=pr)
    return pypianoroll.plot_track(pr, cmap=cmap,
                                  beat_resolution=br, downbeats=db)


def import_one(filename, beat_resolution, binarize=0):
    """Import one file and generate a piano roll."""
    pr = pypianoroll.parse(filename, beat_resolution)

    if binarize:
        pr = pypianoroll.binarize(pr)

    merged = pr.get_merged_pianoroll()

    return merged


class Dataset:
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
        self.data = []
        self.targets = []

    def binarize(self, pr):
        """Binarize piano roll matrix and return it."""
        return np.array([[np.float32(1) if e else np.float32(0)
                         for e in r] for r in pr])

    def quantize(self, obj):
        """Quantize time axis."""
        PPQ = obj.resolution
        for instr in obj.instruments:
            for note in instr.notes:
                note.start = obj.time_to_tick(note.start)/PPQ
                note.end = obj.time_to_tick(note.end)/PPQ

        return obj

    # def pitch_transpose(self, ds, value, timesteps=None):
    #     """Perform pitch transposition."""
    #     if timesteps:
    #         range_timesteps = ds.shape[1]
    #         for t in range(range_timesteps):
    #             d = pd.DataFrame(ds[:, t, :])
    #             d.shift(value, axis=1, fill_value=0)
    #             ds[:, t, :] = d.values
    #     else:
    #         d = pd.DataFrame(ds)
    #         d.shift(value, axis=1, fill_value=0)
    #         ds = d.values
    #
    #     return ds

    # def generate(self, bs, limit=None, trans=None, name=''):
    #     """Yield a dataset."""
    #     dim = self.dataset[0].shape[0]
    #     batch = bs
    #     epoch = 0
    #     while True:
    #         cnt = 0
    #         epoch += 1
    #         en = 0
    #         c = 0
    #         while en < dim:
    #             en = c + batch
    #             yield (self.dataset[0][c:en], self.dataset[1][c:en])
    #             c += batch
    #             cnt += 1
    #         if trans == 1:
    #             for v in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
    #                 x = self.pitch_transpose(self.dataset[0], v, 1)
    #                 y = self.pitch_transpose(self.dataset[1], v)
    #                 en = 0
    #                 c = 0
    #                 while en < dim:
    #                     en = c + batch
    #                     yield (x[c:en], y[c:en])
    #                     c += batch
    #                     cnt += 1
    #         print("\nEpoch %d finished for %s, %d" % (epoch, name, cnt))
    #         # if limit == 1:
    #         #     break

    def build_dataset(self, name, step, t_step, steps, down, transpose=0,
                      low_lim=21, high_lim=109):
        """Build a dataset."""
        print("Building %s dataset (%d files)" % (name, len(self.midi_list)))

        for m in self.midi_list:

            prt = import_one(str(self.path / m), beat_resolution=self.fs,
                             binarize=0)
            prt = prt[:, low_lim:high_lim]  # Crop piano roll
            prt = self.binarize(prt)

            if down > 1:
                prt = downsample_roll(prt, steps, down)

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
#                     conc = downsample_one(conc, steps, down)
                    data_new.append(conc)

                data = np.array(data_new)

            if t_step > 0:
                target_new = []

                if self.baseline:
                    for c in range(len(target)-t_step+1):
                        conc = np.array([target[c] for x in range(c,
                                                                  c+t_step)])

#                         conc = downsample_one(conc, steps, down)
                        target_new.append(np.concatenate(conc, axis=None))

                else:
                    for c in range(len(target)-t_step+1):
                        conc = np.array([target[x] for x in range(c,
                                                                  c+t_step)])

#                         conc = downsample_one(conc, steps, down)
                        target_new.append(np.concatenate(conc, axis=None))

                target = np.array(target_new)

            self.data.append(data)
            self.targets.append(target)

        self.concatenate_all()

    def concatenate_all(self):
        """Build dataset by concatenating all files."""
        self.data = np.concatenate([x for x in self.data], axis=0)
        self.targets = np.concatenate([x for x in self.targets], axis=0)
        self.dataset = (self.data, self.targets)
        del self.data
        del self.targets


def generate(datasets, bs=64, trans=0):
    """Yield dataset with random order and transposition."""
    length = datasets[0].shape[0]
    randomize = list(range(length))
    transpositions = list(range(-5, 7))

    while True:
        random.shuffle(randomize)
        b = 0

        while True:
            batch = randomize[b:b+bs]
            b += bs

            if datasets[0][batch, :, :].shape[0] == 0:
                break

            if trans:
                trans = random.choice(transpositions)
                for i in range(datasets[0][batch, :, :].shape[1]):
                    datasets[0][batch, i, :] = pitch_trans(datasets[0][batch,
                                                                       i, :],
                                                           trans)
                datasets[1][batch, :] = pitch_trans(datasets[1][batch, :],
                                                    trans)

            x = datasets[0][batch, :, :]
            y = datasets[1][batch, :]

            yield (x, y)


def pitch_trans(data, value):
    """Perform music transposition."""
    data = pd.DataFrame(data)
    data = data.shift(value, axis=1, fill_value=0)
    return np.array(data.values)


# def downsample_one(inst, steps, down):
#     """Downsample timesteps."""
#     cnt = 0
#     array = []

#     for i in range(int(steps/down)):
#         vect = inst[cnt:cnt+down, :].any(axis=0)
#         array.append(vect)
#         cnt += down

#     return np.array(array)

def downsample_roll(pr, steps, down):
    """Downsample timesteps."""
    cnt = 0
    array = []
    for i in range(int(len(pr)/down)):
        array.append(pr[cnt:cnt+down, :].any(axis=0))
        cnt += down
    array.append(pr[cnt:].any(axis=0))  # Last

    return np.array(array)


# def build_baseline(data, target):
#     for i in range(data.shape[0]):
#         for n in range(data.shape[1]):
#             target[i, 88*n:88*(n+1)] = data[i, -1, :]

#     return target
