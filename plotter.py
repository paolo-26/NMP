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
    name = '20200515-170055'
    data = load_file(name)
    print(data)
    metrics = ['F1 score', 'Loss', 'Precision', 'Recall']
    for m, metric in enumerate(['f1', 'loss', 'precision_1', 'recall_1']):
        plt.figure()
        plt.plot(smooth(data[metric]), '-^', markersize=5.5, label='Training')
        plt.plot(smooth(data['val_' + metric]), '-s', markersize=4.5,
                 label='Validation')
        plt.title(metrics[m])
        plt.ylabel(metrics[m])
        plt.xlabel('Epoch')
        plt.grid(which='both')
        plt.xticks(data["epoch"])
        plt.legend()
    plt.show()

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
