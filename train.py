#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main script."""
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import matplotlib.pyplot as plt
import os
import model as mod
import dataset
from dataset import plot_piano_roll
from pathlib import Path

P = Path(__file__).parent.absolute()
FS = 10  # Sampling frequency


def main():
    """Run main script."""
    # Build Keras model.
    model = mod.build_model()
    now = datetime.now()
    logger = TensorBoard(log_dir=P / 'logs' / now.strftime("%Y%m%d-%H%M%S"),
                         write_graph=True, update_freq='epoch')

    # Load midi files.
    midi_list = [x for x in os.listdir(P / "data") if x.endswith('.mid')]
    epochs = 10
    st = 1
    train_list = midi_list[0:25]
    validation_list = midi_list[25:35]
    test_list = midi_list[50:51]
    print("Train list:  ", train_list)
    print("Validation list:  ", validation_list)
    print("Test list:  ", test_list)

    # Create generators.
    train = dataset.DataGenerator(train_list, P / "data",  fs=FS)
    validation = dataset.DataGenerator(validation_list, P / "data",  fs=FS)
    test = dataset.DataGenerator(test_list, P / "data",  fs=FS)

    # Fit the model.
    model.fit(train.generate(step=st), epochs=epochs,
              steps_per_epoch=train.dim,
              validation_data=validation.generate(step=st),
              validation_steps=validation.dim,
              callbacks=[logger])

    # Evaluate the model.
    print("Evaluation on test set:")
    _, prec, rec, f1 = model.evaluate(test.generate(step=st), steps=test.dim)

    # Make predictions.
    predictions = model.predict(test.generate(step=st), steps=test.dim)
    for c, t in enumerate(list(test.generate(step=st, limit=1))):
        test = t[0]
        test = test[:, 0, :]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        plot_piano_roll(dataset.transpose(test), 0, 128, ax1, FS)
        ax1.set_title('Test ' + str(c))

        plot_piano_roll(dataset.transpose(predictions), 0, 128, ax2, FS)
        ax2.set_title('Predictions ' + str(c))
        plt.show()


if __name__ == '__main__':
    main()
