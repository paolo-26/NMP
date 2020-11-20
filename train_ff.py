#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main script used for training."""
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
# from tensorflow.keras.models import load_model
import keras.metrics
from datetime import datetime
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import os
from nmp import model as mod
from nmp import dataset
# from nmp.dataset import pyplot_piano_roll
# from nmp import plotter
from pathlib import Path
import time
import math
# import pypianoroll
# from pypianoroll import Multitrack, Track
# import numpy as np
# import random
# import copy
import tensorflow as tf

# Variables to set.
D = "data/Piano-midi.de"  # Dataset
# D = "data/Nottingham"  # Dataset
# D = "data/JSB Chorales"  # Dataset
st = 10  # Past timesteps
num_ts = 10  # Predicted timesteps
BS = 64  # Batch size
EPOCHS = 2  # Number of epochs
DOWN = 1  # Downsampling factor
FS = 24  # Beat resolution: number of timesteps per beat in the piano roll.
LOAD = 0
TRANS = 0
SAVE = 1
KEYBOARD_SIZE = 64  # Number of pitches in the pianoroll.
NOTES = '-' + 'no-down'
# NOTES = '-' + 'ff'

# Do not change.
if KEYBOARD_SIZE == 49:
    LOW_LIM = 36  # C2
    HIGH_LIM = 85  # C6

if KEYBOARD_SIZE == 64:
    LOW_LIM = 33  # A1
    HIGH_LIM = 97  # C7

if KEYBOARD_SIZE == 88:
    LOW_LIM = 21  # A0
    HIGH_LIM = 109  # C8

NUM_NOTES = HIGH_LIM - LOW_LIM
CROP = [LOW_LIM, HIGH_LIM]  # Crop plots
P = Path(__file__).parent.absolute()
PLOTS = P / 'plots'  # Plots path


def main():
    #  Import MIDI files
    train_list = [x for x in os.listdir(P / D / 'train') if x.endswith('.mid')]
    valid_list = [x for x in os.listdir(P / D / 'valid') if x.endswith('.mid')]
    test_list = [x for x in os.listdir(P / D / 'test') if x.endswith('.mid')]

    print("\nTrain list:  ", train_list)
    print("\nValidation list:  ", valid_list)
    print("\nTest list:  ", test_list)

    start = time.time()

    train = dataset.Dataset(train_list, P / D / 'train',  fs=FS, bl=0)
    validation = dataset.Dataset(valid_list, P / D / 'valid',  fs=FS, bl=0)
    test = dataset.Dataset(test_list, P / D / 'test',  fs=FS, bl=0)

    train.build_dataset("training", step=st, t_step=num_ts, steps=st,
                        down=DOWN, low_lim=LOW_LIM, high_lim=HIGH_LIM)
    validation.build_dataset("validation", step=st, t_step=num_ts, steps=st,
                             down=DOWN, low_lim=LOW_LIM, high_lim=HIGH_LIM)
    test.build_dataset("test", step=st, t_step=num_ts, steps=st,
                       down=DOWN, low_lim=LOW_LIM, high_lim=HIGH_LIM)

    end = time.time()
    print("Done\nLoading time: %.2f" % (end-start))

    # Build Keras the model
    model = mod.build_model((st, NUM_NOTES), (num_ts), NUM_NOTES, BS)
    mod.compile_model(model, 'binary_crossentropy', 'adam',
                      metrics=['accuracy',
                               mod.f1,
                               keras.metrics.Precision(),
                               keras.metrics.Recall()])

    model.summary()

    now = datetime.now()
    # Save logs
    logger = TensorBoard(log_dir=P / 'logs' / now.strftime("%Y%m%d-%H%M%S"),
                         write_graph=True, update_freq='epoch')

    csv_logger = CSVLogger(P / 'logs' / (now.strftime("%Y%m%d-%H%M%S") + '-' +
                           str(st) + '-' + str(num_ts) + '.csv'),
                           separator=',', append=False)

    # Checkpoints
    checkpoint_dir = P / ('models/training_checkpoints/' +
                          now.strftime("%Y%m%d-%H%M%S"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_prefix,
                        save_weights_only=True,
                        save_best_only=True,
                        period=1)

    # Fit the model (generator).
    epochs = EPOCHS
    start = time.time()
    size_train = math.ceil(train.dataset[0].shape[0] / BS)
    spe_train = size_train
    size_valid = math.ceil(validation.dataset[0].shape[0] / BS)
    spe_valid = size_valid
    print("Train dataset shape: ", train.dataset[0].shape, "\n")
    print("Train dataset target shape: ", train.dataset[1].shape, "\n")
    history = model.fit(dataset.generate((train.dataset[0],
                                          train.dataset[1]),
                        trans=1),
                        epochs=epochs,
                        steps_per_epoch=spe_train,
                        validation_data=dataset.generate((validation.dataset[0],
                                                          validation.dataset[1]
                                                          )),
                        validation_steps=spe_valid,
                        callbacks=[logger, csv_logger, checkpoint_callback])
    end = time.time()
    print("\nTraining time: ", (end-start), "\n")

    hist = pd.DataFrame(history.history)
    # Loss.
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))
    plt.plot(hist['loss'], '-', c='tab:red', label='Train', ms=8, alpha=0.8)
    plt.plot(hist['val_loss'], '-', c='tab:orange', label='Validation', ms=8, alpha=0.8)
    plt.xlabel('Epoch', fontsize='x-large')
    plt.legend(fontsize='x-large')
    plt.title('Loss: Binary cross-entropy', fontsize='x-large')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax = plt.gca()
    ax.tick_params(labelsize='x-large')

    # F1 score.
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))
    plt.plot(hist['f1'], 's-', c='tab:blue', label='Train', ms=8, alpha=0.8)
    plt.plot(hist['val_f1'], 'o-', c='tab:orange', label='Validation', ms=8,
             alpha=0.8)
    plt.xlabel('Epoch', fontsize='x-large')
    plt.legend(fontsize='x-large')
    plt.title('F1 Score', fontsize='x-large')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax = plt.gca()
    ax.tick_params(labelsize='x-large')

    # Recall.
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))
    plt.plot(hist['recall_1'], 's-', c='tab:blue', label='Train', ms=8, alpha=0.8)
    plt.plot(hist['val_recall_1'], 'o-', c='tab:orange', label='Validation', ms=8, alpha=0.8)
    plt.xlabel('Epoch', fontsize='x-large')
    plt.legend(fontsize='x-large')
    plt.title('Recall', fontsize='x-large')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax = plt.gca()
    ax.tick_params(labelsize='x-large')

    # Precision
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))
    plt.plot(hist['precision_1'], 's-', c='tab:blue', label='Train', ms=8, alpha=0.8)
    plt.plot(hist['val_precision_1'], 'o-', c='tab:orange', label='Validation', ms=8, alpha=0.8)
    plt.xlabel('Epoch', fontsize='x-large')
    plt.legend(fontsize='x-large')
    plt.title('Precision', fontsize='x-large')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax = plt.gca()
    ax.tick_params(labelsize='x-large')

    if SAVE:
        hist['loss'].to_csv(P / 'tables2' / ('ff-' + D[5:] + '-loss-train-' +
                                            str(BS) + NOTES + '.dat').lower(),
                            sep=' ', header=None)
        hist['val_loss'].to_csv(P / 'tables2' / ('ff-' + D[5:] + '-loss-valid-'
                                            + str(BS) + NOTES + '.dat').lower(),
                                sep=' ', header=None)

        hist['f1'].to_csv(P / 'tables2' / ('ff-' + D[5:] + '-f1-train-' + str(BS) + NOTES + '.dat').lower(), sep=' ', header=None)
        hist['val_f1'].to_csv(P / 'tables2' / ('ff-' + D[5:] + '-f1-valid-' + str(BS) + NOTES + '.dat').lower(), sep=' ', header=None)

        hist['precision_1'].to_csv(P / 'tables2' / ('ff-' + D[5:] + '-precision-train-' + str(BS) + NOTES + '.dat').lower(), sep=' ', header=None)
        hist['val_precision_1'].to_csv(P / 'tables2' / ('ff-' + D[5:] + '-precision-valid-' + str(BS) + NOTES + '.dat').lower(), sep=' ', header=None)

        hist['recall_1'].to_csv(P / 'table2s' / ('ff-' + D[5:] + '-recall-train-' + str(BS) + NOTES + '.dat').lower(), sep=' ', header=None)
        hist['val_recall_1'].to_csv(P / 'tables2' / ('ff-' + D[5:] + '-recall-valid-' + str(BS) + NOTES + '.dat').lower(), sep=' ', header=None)

    print("Tables are saved!")

    plt.show()
    # END ####################################################################.
    # model = mod.build_model((st, NUM_NOTES), (num_ts), NUM_NOTES, BS)
    # mod.compile_model(model, 'binary_crossentropy', 'adam',
    #                   metrics=['accuracy', mod.f1, keras.metrics.Precision(), keras.metrics.Recall()])
    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    # model.build(tf.TensorShape([1, None]))

    # # %% [markdown]
    # # ### Model evaluation

    # # %%
    # print("Evaluation on train set:")
    # e_train = model.evaluate(x=train.dataset[0],
    #                          y=train.dataset[1],
    #                          batch_size=BS)

    # print("\nEvaluation on validation set:")
    # e_valid = model.evaluate(x=validation.dataset[0],
    #                          y=validation.dataset[1],
    #                          batch_size=BS)

    # print("\nEvaluation on test set:")
    # e_test = model.evaluate(x=test.dataset[0],
    #                         y=test.dataset[1],
    #                         batch_size=BS)

    # results = {out: e_train[i] for i, out in enumerate(model.metrics_names)}
    # res = pd.DataFrame(list(results.items()), columns=['metric', 'train'])
    # res = res.set_index('metric')

    # results2 = {out: e_valid[i] for i, out in enumerate(model.metrics_names)}
    # res2 = pd.DataFrame(list(results2.items()), columns=['metric', 'validation'])
    # res2 = res2.set_index('metric')

    # results3 = {out: e_test[i] for i, out in enumerate(model.metrics_names)}
    # res3 = pd.DataFrame(list(results3.items()), columns=['metric', 'test'])
    # res3 = res3.set_index('metric')

    # result = pd.concat([res, res2, res3], axis=1, sort=False)
    # result

    # # %% [markdown]
    # # ### Make predictions
    # # Predictions from test dataset

    # # %%
    # L = test.dataset[0].shape[0]
    # # L -= L % BS
    # predictions = model.predict(x=test.dataset[0][:L, :, :])
    # predictions_bin = dataset.threshold(predictions)
    # # print("Pred shape: ", predictions.shape)
    # # predictions = predictions[:, 88*0:88*1]  # First timestep
    # # print("Test shape: ", test.dataset[1].shape, "\n\n\n")
    # # test2 = test.dataset[1][:, :88]  # First timestep
    # # prediction_new = dataset.transpose(predictions)
    # # prediction_new = dataset.convert(prediction_new)
    # # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # # plot_piano_roll(dataset.transpose(test2), 21, 109, ax1, FS)
    # # ax1.set_title('Test  target')

    # # plot_piano_roll(prediction_new, 21, 109, ax2, FS)
    # # ax2.set_title('Test predictions')

    # pyplot_piano_roll(test.dataset[1][:, :NUM_NOTES],
    #                   cmap="Greens", low_lim=LOW_LIM, high_lim=HIGH_LIM)
    # plt.title("Test target (ground truth)")
    # plt.ylim(CROP)

    # pyplot_piano_roll(predictions[:, :NUM_NOTES],
    #                   cmap="Purples", low_lim=LOW_LIM, high_lim=HIGH_LIM)
    # plt.title("Test predictions (not thresholded)")
    # plt.ylim(CROP)

    # # %% [markdown]
    # # ### Evaluate AUC - ROC
    # # Evaluate metric on predictions and baseline with respect to the ground truth of test dataset

    # # %%
    # # Build baseline
    # if D == "data/JSB-Chorales-dataset":
    #     baseline = dataset.Dataset(test_list, P / D / 'test',  fs=FS, bl=1, quant=Q)
    #     baseline.build_choral("test", step=st, t_step=num_ts, steps=st,
    #                        low_lim=LOW_LIM, high_lim=HIGH_LIM)

    # else:
    #     baseline = dataset.Dataset(test_list, P / D / 'test',  fs=FS, bl=1, quant=Q)
    #     baseline.build_dataset("baseline", step=st, t_step=num_ts, steps=st,
    #                            down=DOWN, low_lim=LOW_LIM, high_lim=HIGH_LIM)

    # print("")
    # print("Baseline shape: ", baseline.dataset[1].shape)
    # print("Test shape: ", test.dataset[1].shape)

    # pred_auc = ev_metrics.compute_auc(test.dataset[1][:L, :], predictions, NUM_NOTES)
    # base_auc = ev_metrics.compute_auc(test.dataset[1][:L, :], baseline.dataset[1][:L, :], NUM_NOTES)
    # # pred_auc = ev_metrics.compute_auc(test.dataset[1], predictions, NUM_NOTES)
    # # base_auc = ev_metrics.compute_auc(test.dataset[1], baseline.dataset[1], NUM_NOTES)

    # # %%
    # fig, (ax1, ax2, axcb) = plt.subplots(1, 3, constrained_layout=True,
    #                                      figsize=(8, 8),
    #                                      gridspec_kw={'width_ratios':[1, 1, 0.08]})
    # g1 = sns.heatmap(pred_auc, vmin=0.5, vmax=1, cmap='copper', ax=ax1, cbar=False)
    # g1.set_ylabel('')
    # g1.set_xlabel('')
    # g1.set_yticklabels(g1.get_yticklabels(), rotation=0)
    # ax1.set_xlabel('Time (step)')
    # ax1.set_ylabel('Pitch')
    # ax1.set_title('AUC-ROC (prediction)')
    # g2 = sns.heatmap(base_auc, vmin=0.5, vmax=1, cmap='copper', ax=ax2, cbar_ax=axcb)
    # g2.set_ylabel('')
    # g2.set_xlabel('')
    # g2.set_yticks([])
    # ax2.set_xlabel('Time (step)')
    # ax2.set_title('AUC-ROC (baseline)')
    # ax1.get_shared_y_axes().join(ax1,ax2)
    # plt.savefig(PLOTS / 'heat.eps', format='eps')
    # print(pred_auc.shape)

    # # %%
    # c1 = 0
    # c2 = 88
    # fig, (ax1, ax2, axcb) = plt.subplots(1, 3, constrained_layout=True,
    #                                      figsize=(8, 6),
    #                                      gridspec_kw={'width_ratios':[1, 1, 0.08]})
    # g1 = sns.heatmap(pred_auc[c1:c2], vmin=0.5, vmax=1, cmap='gray', ax=ax1, cbar=False)
    # g1.set_ylabel('')
    # g1.set_xlabel('')
    # g1.set_yticklabels(g1.get_yticklabels(), rotation=0)
    # ax1.set_xlabel('Time (step)')
    # ax1.set_ylabel('Pitch')
    # ax1.set_title('AUC-ROC (crop) [prediction]')
    # g2 = sns.heatmap(base_auc[c1:c2], vmin=0.5, vmax=1, cmap='gray', ax=ax2, cbar_ax=axcb)
    # g2.set_ylabel('')
    # g2.set_xlabel('')
    # g2.set_yticks([])
    # ax2.set_xlabel('Time (step)')
    # ax2.set_title('AUC-ROC (crop) [baseline]')
    # ax1.get_shared_y_axes().join(ax1,ax2)
    # plt.savefig(PLOTS / 'heat_crop.eps', format='eps')
    # print(pred_auc.shape)

    # # %%
    # fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 4))

    # ax.plot(range(1, num_ts + 1), np.mean(pred_auc[c1:c2]), 'x', c='tab:blue', label='prediction', ms=10)
    # ax.plot(range(1, num_ts + 1), np.mean(base_auc[c1:c2]), 'o', c='tab:green', label='baseline ', ms=7)

    # ax.set_ylim([0.4, 1])
    # ax.set_ylim([0.4, 1])
    # ax.legend()
    # plt.title('Avg. AUC-ROC per predicted timestep')
    # plt.xlabel('Timestep')
    # # plt.xticks([0, 2, 4, 6, 8, 10])
    # plt.ylabel('ROC AUC')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # name = 'auc' + str()
    # plt.grid()
    # # plt.savefig(PLOTS / 'aucDe222.eps', format='eps')
    # # fig.savefig(PLOTS / 'comp-ff-auc.pdf')
    # print("Predict. mean value:", np.mean(np.mean(pred_auc[c1:c2])))
    # print("Baseline mean value:", np.mean(np.mean(base_auc[c1:c2])))

    # # %%
    # np.mean(pred_auc[c1:c2])

    # # %%
    # np.mean(base_auc[c1:c2])

    # # %%
    # # auc_df = pd.DataFrame(
    # #     {'pred': np.mean(pred_auc[c1:c2]),
    # #      'base': np.mean(base_auc[c1:c2])})
    # # auc_df['pred'].to_csv(('tables/ff-' + D[5:] + '-auc-pred-' + str(BS) + NOTES + '.dat').lower(), sep=' ', header=None)
    # # auc_df['base'].to_csv(('tables/ff-' + D[5:] + '-auc-base-' + str(BS) + NOTES + '.dat').lower(), sep=' ', header=None)

    # # %% [markdown]
    # # ### Piano rolls
    # # - test data (input of the network)
    # #
    # # - test target (ground truth)
    # #
    # # - model predictions (output of the network)
    # #
    # # - baseline (repetition of  the last input)

    # # %%
    # t=0  # Timestep to visualize
    # plt.rcParams["figure.figsize"] = (10, 4)
    # pyplot_piano_roll(test.dataset[0][:, 0, :],
    #                   cmap="Blues", low_lim=LOW_LIM, high_lim=HIGH_LIM)
    # plt.title("Test data (input)")
    # plt.ylim(CROP)
    # # plt.savefig(PLOTS / ('pr' + str(t) + 'data.png'))

    # pyplot_piano_roll(predictions[:, NUM_NOTES*t:NUM_NOTES*(t+1)],
    #                   cmap="Purples", low_lim=LOW_LIM, high_lim=HIGH_LIM)
    # plt.title("Predictions")
    # plt.ylim(CROP)
    # # plt.savefig(PLOTS / ('pr' + str(t) + 'pred.png'))

    # pyplot_piano_roll(test.dataset[1][:, NUM_NOTES*t:NUM_NOTES*(t+1)],
    #                   cmap="Greens", low_lim=LOW_LIM, high_lim=HIGH_LIM)
    # plt.title("Test target (ground truth)")
    # plt.ylim(CROP)
    # # plt.savefig(PLOTS / ('pr' + str(t) + 'target.png'))

    # pyplot_piano_roll(baseline.dataset[1][:, NUM_NOTES*t:NUM_NOTES*(t+1)],
    #                   cmap="Reds", low_lim=LOW_LIM, high_lim=HIGH_LIM)
    # plt.title("Baseline")
    # plt.ylim(CROP)
    # # plt.savefig(PLOTS / ('pr' + str(t) + 'base.png'))

    # # %%
    # t=0  # Timestep to visualize
    # plt.rcParams["figure.figsize"] = (10, 4)
    # pyplot_piano_roll(predictions[:, NUM_NOTES*t:NUM_NOTES*(t+1)],
    #                   cmap="Greys", low_lim=LOW_LIM, high_lim=HIGH_LIM)
    # plt.title("Predictions")
    # plt.ylim([50, 80])
    # # plt.savefig(PLOTS / ('pr' + str(t) + 'predn.png'))

    # t=4  # Timestep to visualize
    # pyplot_piano_roll(predictions[:, NUM_NOTES*t:NUM_NOTES*(t+1)],
    #                   cmap="Greys", low_lim=LOW_LIM, high_lim=HIGH_LIM)
    # plt.title("Predictions")
    # plt.ylim([50, 80])
    # # plt.savefig(PLOTS / ('pr' + str(t) + 'predn.png'))

    # t=7  # Timestep to visualize
    # pyplot_piano_roll(predictions[:, NUM_NOTES*t:NUM_NOTES*(t+1)],
    #                   cmap="Greys", low_lim=LOW_LIM, high_lim=HIGH_LIM)
    # plt.title("Predictions")
    # plt.ylim([50, 80])
    # # plt.savefig(PLOTS / ('pr' + str(t) + 'predn.png'))


if __name__ == "__main__":
    main()
