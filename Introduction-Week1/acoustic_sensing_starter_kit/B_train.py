#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script for the "Acoustic Sensing Starter Kit"
[Zöller, Gabriel, Vincent Wall, and Oliver Brock. “Active Acoustic Contact Sensing for Soft Pneumatic Actuators.” In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.]

This script _trains_ a KNN classifier to predict the previously recorded data samples.

In 'USER SETTINGS' define:
BASE_DIR - path where data is read from
MODEL_NAME - name of the sensor model. used as folder name.
TEST_SIZE - ratio of samples left out for testing. leave at '0' to use all samples for training.

@author: Vincent Wall, Gabriel Zöller
@copyright 2020 Robotics and Biology Lab, TU Berlin
@licence: BSD Licence
"""

import numpy
import librosa
import os
import pandas
import pickle

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

from A_record import MODEL_NAME

# ==================
# USER SETTINGS
# ==================
BASE_DIR = "."
SENSORMODEL_FILENAME = "sensor_model.pkl"
TEST_SIZE = (
    0  # percentage of samples left out of training and used for reporting test score
)
SHOW_PLOTS = True
# ==================

SR = 48000
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


def get_num_and_label(filename):
    try:
        # remove file extension
        name = os.path.splitext(filename)[0]
        # remove initial number
        name = name.split("_")
        num = int(name[0])
        label = "_".join(name[1:])
        return num, label
    except ValueError:
        # filename with different formatting. ignore.
        return -1, None


def load_sounds(path):
    """Load soundfiles from disk"""
    filenames = sorted(os.listdir(path))
    sounds = []
    labels = []
    for fn in filenames:
        n, label = get_num_and_label(fn)
        if n < 0:
            # filename with different formatting. ignore.
            continue
        elif n == 0:
            # zero index contains active sound
            global SOUND_NAME
            SOUND_NAME = label
        else:
            sound = librosa.load(os.path.join(path, fn), sr=SR)[0]
            sounds.append(sound)
            labels.append(label)
    print(f"Loaded **{len(sounds)}** sounds with \nlabels: {sorted(set(labels))}")
    return sounds, labels


def sound_to_spectrum(sound):
    """Convert sounds to frequency spectra"""
    spectrum = numpy.fft.rfft(sound)
    amplitude_spectrum = numpy.abs(spectrum)
    d = 1.0 / SR
    freqs = numpy.fft.rfftfreq(len(sound), d)
    index = pandas.Index(freqs)
    series = pandas.Series(amplitude_spectrum, index=index)
    return series


def sound_to_spectrum_stft(sound, n_fft=4096, in_dB=False):
    spectrogram = numpy.abs(librosa.stft(sound, n_fft=n_fft))
    spectrum = spectrogram.sum(axis=1)
    if in_dB:
        # convert to decibel scale
        spectrum = librosa.amplitude_to_db(spectrum, ref=numpy.max)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    index = pandas.Index(freqs)
    series = pandas.Series(spectrum, index=index)
    return series


def save_sensor_model(path, clf, filename):
    """Saves sensor model to disk"""
    with open(os.path.join(path, filename), "wb") as f:
        pickle.dump(clf, f, protocol=PICKLE_PROTOCOL)


def plot_spectra(spectra, labels):
    from matplotlib import pyplot

    fig, ax = pyplot.subplots(1)
    color_list = pyplot.rcParams["axes.prop_cycle"].by_key()["color"]
    cdict = dict(zip(sorted(list(set(labels))), color_list))
    for i, (s, l) in enumerate(zip(spectra, labels)):
        ax.plot(s, c=cdict[l])

    from matplotlib.lines import Line2D

    legend_lines = [Line2D([0], [0], color=col, lw=4) for col in cdict.values()]
    legend_labels = list(cdict.keys())
    ax.legend(legend_lines, legend_labels)

    fig.show()


def main():
    print("Running for model '{}'".format(MODEL_NAME))
    global DATA_DIR
    DATA_DIR = os.path.join(BASE_DIR, MODEL_NAME)

    sounds, labels = load_sounds(DATA_DIR)
    # spectra = [sound_to_spectrum(sound) for sound in sounds]
    spectra = [sound_to_spectrum_stft(sound) for sound in sounds]
    classes = list(set(labels))

    if SHOW_PLOTS:
        plot_spectra(spectra, labels)

    if TEST_SIZE > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            spectra, labels, test_size=TEST_SIZE
        )
    else:
        X_train, y_train = (spectra, labels)

    clf = KNeighborsClassifier()  # using default KNN classifier
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    print("Fitted sensor model to data!")
    print("Training score: {:.2f}".format(train_score))

    if TEST_SIZE > 0:
        test_score = clf.score(X_test, y_test)
        print("Test score: {:.2f}".format(test_score))

    save_sensor_model(DATA_DIR, clf, SENSORMODEL_FILENAME)
    print("\nSaved model to '{}'".format(os.path.join(DATA_DIR, SENSORMODEL_FILENAME)))

    if SHOW_PLOTS:
        pyplot.pause(0.1)
        pyplot.show()


if __name__ == "__main__":
    main()
