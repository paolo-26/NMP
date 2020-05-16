#!/usr/bin/env python
"""Plotter."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
P = Path(__file__).parent.absolute()


def smooth(scalars, weight=0.4):
    """Smooth values like TensorBoard."""
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def load_file(filename):
    """Load csv file."""
    df = pd.read_csv(P / 'logs' / filename / 'log.csv')
    return df


def main():
    """Plot training and validation data."""

    name = '20200516-131302'  # 20 min Adam 1-1
    # name = '20200516-115508'  # 10 epochs SGD 1-1
    data = load_file(name)
    print(data)
    metrics = ['F1 score', 'Loss', 'Precision', 'Recall']
    for m, metric in enumerate(['f1', 'loss', 'precision_1', 'recall_1']):
        plt.figure(constrained_layout=True, figsize=(6, 4))
        plt.plot(smooth(data[metric]), '-', markersize=5.5, label='Training')
        plt.plot(smooth(data['val_' + metric]), 'o', markersize=5,
                 label='Validation')
        plt.title(metrics[m], fontsize='x-large')
        plt.ylabel(metrics[m], fontsize='x-large')
        plt.xlabel('Epoch', fontsize='x-large')
        # plt.grid(which='both')
        plt.xticks([0, 5, 10, 15, 20])
        # plt.xticks(data["epoch"])
        plt.legend(fontsize='x-large')
        ax = plt.gca()
        ax.tick_params(labelsize='x-large')
        plt.savefig(P / 'plots' / (name + '-' + metric + '.eps'),
                    format='eps')
    # plt.show()

    # labels = ['Training', 'Validation']
    # metrics = ['F1 score', 'Loss', 'Precision', 'Recall']
    # for m, metric in enumerate(['f1', 'loss']):
    #     plt.figure()
    #     for d, dset in enumerate(['train', 'validation']):
    #         data = load_file('run-' + name + '_' + dset + '-tag-epoch_' +
    #                          metric + '.csv')
    #         print(data)
    #         data["Value"] = smooth(data["Value"], 0.6)
    #         plt.plot(data["Step"], data["Value"], '-o', markersize=3.5,
    #                  label=labels[d])
    #     plt.grid(which='both')
    #     plt.title(metrics[m])
    #     plt.ylabel(metrics[m])
    #     plt.xlabel('Epoch')
    #     plt.xticks(data["Step"])
    #     plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
