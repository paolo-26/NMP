#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main script."""
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from datetime import datetime
import matplotlib.pyplot as plt
import os
import model as mod
import dataset
from dataset import plot_piano_roll
from pathlib import Path

P = Path(__file__).parent.absolute()
FS = 100  # Sampling frequency


def main():
    """Run main script."""
    # Load midi files.
    midi_list = [x for x in os.listdir(P / "data") if x.endswith('.mid')]
    epochs = 20
    st = 1
    num_ts = 1

    # All dataset
    # train_list = midi_list[0:200]
    # validation_list = midi_list[201:231]
    # test_list = midi_list[232:233]

    # Small dataset
    train_list = midi_list[0:25]
    validation_list = midi_list[25:32]
    test_list = midi_list[40:43]
    print("Train list:  ", train_list)
    print("Validation list:  ", validation_list)
    print("Test list:  ", test_list)

    # Build Keras model.
    model = mod.build_model(st*88, num_ts)
    now = datetime.now()
    logger = TensorBoard(log_dir=P / 'logs' / now.strftime("%Y%m%d-%H%M%S"),
                         write_graph=True, update_freq='epoch')

    csv_logger = CSVLogger(P / 'logs' / now.strftime("%Y%m%d-%H%M%S") /
                           'log.csv',
                           separator=',', append=False)
    # Create generators.
    train = dataset.DataGenerator(train_list, P / "data",  fs=FS)
    validation = dataset.DataGenerator(validation_list, P / "data",  fs=FS)
    test = dataset.DataGenerator(test_list, P / "data",  fs=FS)

    # Fit the model.
    model.fit(train.generate(step=st), epochs=epochs,
              steps_per_epoch=train.dim,  shuffle=True,
              validation_data=validation.generate(step=st),
              validation_steps=validation.dim,
              callbacks=[logger, csv_logger])

    # Evaluate the model.
    print("Evaluation on test set:")
    _, prec, rec, f1 = model.evaluate(test.generate(step=st), steps=test.dim)

    for c, t in enumerate(list(test.generate(step=st, limit=1))):
        predictions = model.predict(t)
        predictions = dataset.transpose(predictions)
        predictions = dataset.convert(predictions)
        test = t[0]
        # test = test[:, 0, :]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        plot_piano_roll(dataset.transpose(test), 21, 109, ax1, FS)
        ax1.set_title('Test ' + str(c))

        plot_piano_roll(predictions, 21, 109, ax2, FS)
        ax2.set_title('Predictions ' + str(c))
        # plt.show()

    plt.show()


if __name__ == '__main__':
    main()
