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
    # df = pd.read_csv(P / 'logs' / filename / 'log.csv')
    df = pd.read_csv(P / 'logs' / (filename + '.csv'))
    return df


def main():
    """Plot training and validation data."""

    # exp1
    # name = '20200517-114951-1-1'
    # name = '20200517-115539-2-1'

    # name = '20200517-133959-1-1'
    # name = '20200517-133437-2-1'
    # name = '20200517-132921-3-1'
    # name = '20200517-132246-4-1'
    # name = '20200517-125637-5-1'
    # name = '20200517-142537-6-1'

    # Batches.
    # name = '20200518-094510-1-1'
    # name = '20200518-095410-1-1'
    # name = '20200518-100240-1-1'
    # name = '20200518-100949-1-1'

    # More timesteps
    # name = '20200517-170927-2-1'
    # name = '20200517-170358-2-2'
    # name = '20200517-165712-2-3'
    # name = '20200517-164007-2-4'
    # name = '20200517-163431-2-5'
    # name = '20200517-164558-2-6'

    # name = '20200602-105028-1-1'  # Baseline 1-1
    # name = '20200602-112257-1-1'  # 1-1
    # data = load_file(name)
    # print(data)
    # metrics = ['F1 score', 'Loss', 'Precision', 'Recall']
    # for m, metric in enumerate(['f1', 'loss', 'precision_1', 'recall_1']):
    #     plt.figure(constrained_layout=True, figsize=(6, 4))
    #     plt.plot(smooth(data[metric]), '-', markersize=5.5, label='Training')
    #     plt.plot(smooth(data['val_' + metric]), 'o', markersize=5,
    #              label='Validation')
    #     plt.title(metrics[m], fontsize='x-large')
    #     plt.ylabel(metrics[m], fontsize='x-large')
    #     plt.xlabel('Epoch', fontsize='x-large')
    #     # plt.grid(which='both')
    #     plt.xticks([0, 5, 10, 15, 20])
    #     # plt.xticks(data["epoch"])
    #     if metric == 'f1':
    #         plt.ylim([0, 1])
    #     plt.legend(fontsize='x-large')
    #     ax = plt.gca()
    #     ax.tick_params(labelsize='x-large')
    #     plt.savefig(P / 'plots' / (name + '-' + metric + '.eps'),
    #                 format='eps')
    #     plt.savefig(P / 'plots' / (name + '-' + metric + '.png'),
    #                 format='png')

    # Single  plots.
    file = ['20200517-170927-2-1', '20200517-170358-2-2',
            '20200517-165712-2-3', '20200517-164007-2-4',
            '20200517-163431-2-5', '20200517-164558-2-6']
    metrics = ['Loss']
    styles = ['s', 'o', '^', 'd', 'p', 'P']
    plt.figure(constrained_layout=True, figsize=(6, 4))
    for n, f in enumerate(file):
        data = load_file(f)
        for m, metric in enumerate(['loss']):
            plt.plot(smooth(data['val_' + metric]), '-'+styles[n],
                     markersize=6, label=n+1)
            plt.title(metrics[m], fontsize='x-large')
            plt.ylabel(metrics[m], fontsize='x-large')
            plt.xlabel('Epoch', fontsize='x-large')
            plt.xticks([0, 5, 10, 15])
            plt.legend(fontsize='x-large', ncol=3)
            ax = plt.gca()
            ax.tick_params(labelsize='x-large')
            plt.savefig(P / 'plots' / ('Lossmore.eps'),
                        format='eps')
    plt.show()

    # Other
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
