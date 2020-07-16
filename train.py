#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main script used for training."""
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import model as mod
import dataset
from dataset import pyplot_piano_roll
from pathlib import Path
from plotter import smooth
import time
import math
import ev_metrics
import pypianoroll
from pypianoroll import Multitrack, Track
import numpy as np

# P = Path(__file__).parent.absolute()
P = Path(os.path.abspath(''))  # Compatible with Jupyter Notebook

PLOTS = P / 'plots'  # Plots path
FS = 10  # Sampling frequency. 10 Hz = 100 ms
Q = 0  # Quantize?
st = 10  # Past timesteps
num_ts = 10  # Predicted timesteps
D = "data"  # Dataset (synth or data)
CROP = [21, 109]  # Crop plots


# ### Generate list of MIDI files

# In[ ]:


# Load midi files.
midi_list = [x for x in os.listdir(P / D) if x.endswith('.mid')]
print("Total number of MIDI files:", len(midi_list))

if D == "data":  # Piano dataset
    train_list = midi_list[0:165]
    validation_list = midi_list[166:213]
    test_list = midi_list[213:236]

if D == "synth":  # Synth dataset
    train_list = midi_list[0:2500]
    validation_list = midi_list[2500:3000]
    test_list = midi_list[3000:4000]

# Small dataset
# train_list = midi_list[0:10]
# validation_list = midi_list[2:3]
# test_list = midi_list[61:65]

# print("\nTrain list:  ", train_list)
# print("\nValidation list:  ", validation_list)
# print("\nTest list:  ", test_list)


# ## Datasets
# ### Generate data from lists
# Training, validation and test sets.

# In[ ]:


train = dataset.DataGenerator(train_list, P / D,  fs=FS, bl=0, quant=Q)
validation = dataset.DataGenerator(validation_list, P / D,  fs=FS, bl=0, quant=Q)
test = dataset.DataGenerator(test_list, P / D,  fs=FS, bl=0, quant=Q)
train.build_dataset("training", step=st, t_step=num_ts)
validation.build_dataset("validation", step=st, t_step=num_ts)
test.build_dataset("test", step=st, t_step=num_ts)
print("Done")


# ### Piano rolls of training dataset
# Input and output piano rolls

# In[ ]:


plt.rcParams["figure.figsize"] = (20, 8)
pyplot_piano_roll(train.dataset[0][:, 0, :])
plt.title("Train data")
plt.ylim(CROP)
pyplot_piano_roll(train.dataset[1][:, :88], cmap="Oranges")
plt.title("Train target")
plt.ylim(CROP)


# ### Data augmentation
# Transposition

# In[ ]:


train.build_transposed()


# ## Keras
# ### Build the model

# In[ ]:


# Build Keras model.
model = mod.build_model((st, 88), num_ts)
mod.compile_model(model, 'binary_crossentropy', 'rmsprop')
model.summary()
now = datetime.now()

# Save logs
logger = TensorBoard(log_dir=P / 'logs' / now.strftime("%Y%m%d-%H%M%S"),
                    write_graph=True, update_freq='epoch')

csv_logger = CSVLogger(P / 'logs' / (now.strftime("%Y%m%d-%H%M%S") + '-' +
                      str(st) + '-' + str(num_ts) + '.csv'),
                      separator=',', append=False)


# ### Fit the model
# Define batch size ```BS``` and number of ```epochs```

# In[ ]:


# Fit the model.

BS = 64  # Batch size
epochs = 15
start = time.time()
# size_train = math.ceil(train.dataset[0].shape[0] / BS)
# spe_train = size_train #+ size_train*10
# size_valid = math.ceil(validation.dataset[0].shape[0] / BS)
# spe_valid = size_valid #+ size_valid*10
print("Train dataset shape: ", train.dataset[0].shape, "\n")
print("Train dataset target shape: ", train.dataset[1].shape, "\n")

# Fit generator. Data should be shuffled before fitting.
# history = model.fit(train.generate(bs=BS, limit=epochs, trans=0, name='train'), epochs=epochs,
#           steps_per_epoch=spe_train,
#           validation_data=validation.generate(bs=BS, limit=epochs, trans=0, name='valid'),
#           validation_steps=spe_valid,
#           shuffle=True,
#           callbacks=[logger, csv_logger])


# Normal fit. Auto-shuffles data.
history = model.fit(x=train.dataset[0], y=train.dataset[1],
                    epochs=epochs, batch_size=BS, shuffle=True,
                    validation_data=(validation.dataset[0],
                                     validation.dataset[1]),
                    callbacks=[logger, csv_logger])

end = time.time()


# ### History
# 
# ```f1_first```: F1-score on first predicted timestep
# 
# ```f1_last```: F1-score on last predicted timestep

# In[ ]:


print("\nTraining time: ", (end-start), "\n")
hist = pd.DataFrame(history.history)
print(hist)


# ### Plot loss function of training and validation sets

# In[ ]:


fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 4))
plt.plot(hist['val_loss'], 'o', c='tab:orange', label='Validation', ms=8, alpha=0.8)
plt.plot(hist['loss'], 'ro-', c='tab:red', label='Train', ms=8, alpha=0.8)
plt.xlabel('Epoch')
plt.xticks(range(epochs))
plt.legend()
plt.title('Loss: Binary cross-entropy')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig(PLOTS / 'loss.eps', fmt='eps')
plt.show()
print("Training time: ", (end-start))


# ### Model evaluation

# In[ ]:


print("Evaluation on train set:")
e_train = model.evaluate(x=train.dataset[0],
                         y=train.dataset[1],
                         batch_size=BS)

print("\nEvaluation on validation set:")
e_valid = model.evaluate(x=validation.dataset[0],
                         y=validation.dataset[1],
                         batch_size=BS)

print("\nEvaluation on test set:")
e_test = model.evaluate(x=test.dataset[0],
                        y=test.dataset[1],
                        batch_size=BS)

results = {out: e_train[i] for i, out in enumerate(model.metrics_names)}
res = pd.DataFrame(list(results.items()), columns=['metric', 'train'])
res = res.set_index('metric')

results2 = {out: e_valid[i] for i, out in enumerate(model.metrics_names)}
res2 = pd.DataFrame(list(results2.items()), columns=['metric', 'validation'])
res2 = res2.set_index('metric')

results3 = {out: e_test[i] for i, out in enumerate(model.metrics_names)}
res3 = pd.DataFrame(list(results3.items()), columns=['metric', 'test'])
res3 = res3.set_index('metric')


result = pd.concat([res, res2, res3], axis=1, sort=False)
result


# ### Make predictions
# Predictions from test dataset

# In[ ]:


predictions = model.predict(x=test.dataset[0])
predictions_bin = dataset.convert(predictions)
# print("Pred shape: ", predictions.shape)
# predictions = predictions[:, 88*0:88*1]  # First timestep
# print("Test shape: ", test.dataset[1].shape, "\n\n\n")
# test2 = test.dataset[1][:, :88]  # First timestep
# prediction_new = dataset.transpose(predictions)
# prediction_new = dataset.convert(prediction_new)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# plot_piano_roll(dataset.transpose(test2), 21, 109, ax1, FS)
# ax1.set_title('Test  target')

# plot_piano_roll(prediction_new, 21, 109, ax2, FS)
# ax2.set_title('Test predictions')

pyplot_piano_roll(test.dataset[1][:, :88], cmap="Greens")
plt.title("Test target (ground truth)")
plt.ylim(CROP)

pyplot_piano_roll(predictions_bin[:, :88], cmap="Purples")
plt.title("Test predictions")
plt.ylim(CROP)


# ### Evaluate AUC - ROC
# Evaluate metric on predictions and baseline with respect to the ground truth of test dataset

# In[ ]:


# Build baseline
baseline = dataset.DataGenerator(test_list, P / D,  fs=FS, bl=1, quant=Q)
baseline.build_dataset("baseline", step=st, t_step=num_ts)

print("")
print("Baseline shape: ", baseline.dataset[1].shape)
print("Test shape: ", test.dataset[1].shape)
print("Prediction shape: ", predictions_bin.shape)

pred_auc = ev_metrics.compute_auc(test.dataset[1], predictions)
base_auc = ev_metrics.compute_auc(test.dataset[1], baseline.dataset[1])

# print("Predictions mean AUC: ", pred_auc)
# print("Predictions (not thresholded) mean AUC: ", pred_auc)
# print("Baseline mean AUC: ", base_auc)


# In[ ]:


fig, (ax1, ax2, axcb) = plt.subplots(1, 3, constrained_layout=True,
                                     figsize=(8, 8),
                                     gridspec_kw={'width_ratios':[1, 1, 0.08]})
g1 = sns.heatmap(pred_auc, vmin=0.5, vmax=1, cmap='copper', ax=ax1, cbar=False)
g1.set_ylabel('')
g1.set_xlabel('')
g1.set_yticklabels(g1.get_yticklabels(), rotation=0)
ax1.set_xlabel('Time (step)')
ax1.set_ylabel('Pitch')
ax1.set_title('AUC-ROC (prediction)')
g2 = sns.heatmap(base_auc, vmin=0.5, vmax=1, cmap='copper', ax=ax2, cbar_ax=axcb)
g2.set_ylabel('')
g2.set_xlabel('')
g2.set_yticks([])
ax2.set_xlabel('Time (step)')
ax2.set_title('AUC-ROC (baseline)')
ax1.get_shared_y_axes().join(ax1,ax2)
plt.savefig(PLOTS / 'heat.eps', format='eps')
print(pred_auc.shape)


# In[ ]:


c1 = 25
c2 = 70
fig, (ax1, ax2, axcb) = plt.subplots(1, 3, constrained_layout=True,
                                     figsize=(8, 6),
                                     gridspec_kw={'width_ratios':[1, 1, 0.08]})
g1 = sns.heatmap(pred_auc[c1:c2], vmin=0.5, vmax=1, cmap='gray', ax=ax1, cbar=False)
g1.set_ylabel('')
g1.set_xlabel('')
g1.set_yticklabels(g1.get_yticklabels(), rotation=0)
ax1.set_xlabel('Time (step)')
ax1.set_ylabel('Pitch')
ax1.set_title('AUC-ROC (crop) [prediction]')
g2 = sns.heatmap(base_auc[c1:c2], vmin=0.5, vmax=1, cmap='gray', ax=ax2, cbar_ax=axcb)
g2.set_ylabel('')
g2.set_xlabel('')
g2.set_yticks([])
ax2.set_xlabel('Time (step)')
ax2.set_title('AUC-ROC (crop) [baseline]')
ax1.get_shared_y_axes().join(ax1,ax2)
plt.savefig(PLOTS / 'heat_crop.eps', format='eps')
print(pred_auc.shape)


# In[ ]:


fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 4))

ax.plot(range(1, num_ts + 1), np.mean(pred_auc[c1:c2]), 'x', c='tab:blue', label='prediction', ms=10)
ax.plot(range(1, num_ts + 1), np.mean(base_auc[c1:c2]), 'o', c='tab:green', label='baseline ', ms=7)

ax.set_ylim([0.4, 1])
ax.set_ylim([0.4, 1])
ax.legend()
plt.title('Avg. AUC-ROC per predicted timestep')
plt.xlabel('Timestep')
# plt.xticks([0, 2, 4, 6, 8, 10])
plt.ylabel('ROC AUC')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
name = 'auc' + str()
plt.savefig(PLOTS / 'auc.eps', format='eps')
plt.show()

print("Predict. mean value:", np.mean(np.mean(pred_auc[c1:c2])))
print("Baseline mean value:", np.mean(np.mean(base_auc[c1:c2])))


# ### Piano rolls
# - test data (input of the network)
# 
# - test target (ground truth)
# 
# - model predictions (output of the network)
# 
# - baseline (repetition of  the last input)

# In[ ]:


t=0  # Timestep to visualize
plt.rcParams["figure.figsize"] = (10, 4)
pyplot_piano_roll(test.dataset[0][:, 0, :], cmap="Blues")
plt.title("Test data (input)")
plt.ylim(CROP)
plt.savefig(PLOTS / ('pr' + str(t) + 'data.png'))

pyplot_piano_roll(predictions[:, 88*t:88*(t+1)], cmap="Purples")
plt.title("Predictions")
plt.ylim(CROP)
plt.savefig(PLOTS / ('pr' + str(t) + 'pred.png'))

pyplot_piano_roll(test.dataset[1][:, 88*t:88*(t+1)], cmap="Greens")
plt.title("Test target (ground truth)")
plt.ylim(CROP)
plt.savefig(PLOTS / ('pr' + str(t) + 'target.png'))

pyplot_piano_roll(baseline.dataset[1][:, 88*t:88*(t+1)], cmap="Reds")
plt.title("Baseline")
plt.ylim(CROP)
plt.savefig(PLOTS / ('pr' + str(t) + 'base.png'))


# In[ ]:


t=0  # Timestep to visualize
plt.rcParams["figure.figsize"] = (10, 4)
pyplot_piano_roll(predictions[:, 88*t:88*(t+1)], cmap="Greys")
plt.title("Predictions")
plt.ylim([50, 80])
plt.savefig(PLOTS / ('pr' + str(t) + 'predn.png'))

t=4  # Timestep to visualize
pyplot_piano_roll(predictions[:, 88*t:88*(t+1)], cmap="Greys")
plt.title("Predictions")
plt.ylim([50, 80])
plt.savefig(PLOTS / ('pr' + str(t) + 'predn.png'))

t=9  # Timestep to visualize
pyplot_piano_roll(predictions[:, 88*t:88*(t+1)], cmap="Greys")
plt.title("Predictions")
plt.ylim([50, 80])
plt.savefig(PLOTS / ('pr' + str(t) + 'predn.png'))


# ## Additional tests
# Piano dataset, cmaj scale,...

# In[ ]:


# midi_list2 = [x for x in os.listdir(P / "data") if x.endswith('.mid')]
# test_new = midi_list2[0:3]
# test = dataset.DataGenerator(test_new, P / "data",  fs=FS, bl=0, quant=Q)
# test.build_dataset("test", step=st, t_step=num_ts)
# print("Done")


# In[ ]:


# plt.rcParams["figure.figsize"] = (20, 8)
# predictions = model.predict(x=test.dataset[0])
# predictions_bin = dataset.convert(predictions)

# print("Test shape: ", test.dataset[0].shape)
# print("Pred shape: ", predictions_bin.shape)

# pyplot_piano_roll(predictions_bin[:, :88], cmap="Purples")
# plt.title("Predictions")
# plt.ylim(CROP)

# pyplot_piano_roll(test.dataset[1][:, :88], cmap="Greens")
# plt.title("Test target (ground truth)")
# plt.ylim(CROP)


# In[ ]:


# # Build baseline
# baseline = dataset.DataGenerator(test_new, P / "data",  fs=FS, bl=1, quant=Q)
# baseline.build_dataset("baseline", step=st, t_step=num_ts)
# print("")
# print("Baseline shape: ", baseline.dataset[1].shape)
# print("Test shape: ", test.dataset[1].shape)
# print("Prediction shape: ", predictions_bin.shape)
# print("--- --- ---")
# pred_auc2 = ev_metrics.compute_auc(test.dataset[1], predictions)
# base_auc2 = ev_metrics.compute_auc(test.dataset[1], baseline.dataset[1])


# In[ ]:


# fig, (ax1, ax2, axcb) = plt.subplots(1, 3, constrained_layout=True,
#                                      figsize=(12, 8),
#                                      gridspec_kw={'width_ratios':[1, 1, 0.08]})
# g1 = sns.heatmap(pred_auc2[c1:c2], vmin=0.5, vmax=1, cmap='gray', ax=ax1, cbar=False)
# g1.set_ylabel('')
# g1.set_xlabel('')
# ax1.set_xlabel('Time (step)')
# ax1.set_ylabel('Pitch')
# ax1.set_title('AUC-ROC (prediction)')
# g2 = sns.heatmap(base_auc2[c1:c2], vmin=0.5, vmax=1, cmap='gray', ax=ax2, cbar_ax=axcb)
# g2.set_ylabel('')
# g2.set_xlabel('')
# g2.set_yticks([])
# ax2.set_xlabel('Time (step)')
# ax2.set_title('AUC-ROC (baseline)')
# ax1.get_shared_y_axes().join(ax1,ax2)

# print(pred_auc.shape)


# In[ ]:


# fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 4))

# ax.plot(range(1, num_ts + 1), np.mean(pred_auc2[c1:c2]), 'x', c='tab:blue', label='prediction', ms=10)
# ax.plot(range(1, num_ts + 1), np.mean(base_auc2[c1:c2]), 'o', c='tab:green', label='baseline ', ms=7)

# ax.set_ylim([0.4, 1])
# ax.set_ylim([0.4, 1])
# ax.legend()
# plt.title('Avg. AUC-ROC (crop) per predicted timestep')
# plt.xlabel('Timestep')
# # plt.xticks([0, 2, 4, 6, 8, 10])
# plt.ylabel('ROC AUC')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# name = 'auc' + str()
# # plt.savefig(PLOTS / 'auc.eps', format='eps')
# plt.show()

# print("Predict. mean value:", np.mean(np.mean(pred_auc[c1:c2])))
# print("Baseline mean value:", np.mean(np.mean(base_auc[c1:c2])))


# ## Save model to file

# In[ ]:


mod.save_model(model, P / 'model')


# In[ ]:


# Load model.
# my_model = mod.load_model(P / 'model')

