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
import json

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot

from A_record import MODEL_NAME
import preprocessing

# ==================
# USER SETTINGS
# ==================
BASE_DIR = "."
SENSORMODEL_FILENAME = "sensor_model.pkl"
TEST_SIZE = (
    0.2  # percentage of samples left out of training and used for reporting test score
)
SHOW_PLOTS = True
# ==================

SR = 48000

PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
SOUND_NAME = None
DATA_DIR = None


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
            sound = preprocessing.load_audio(os.path.join(path, fn), sr=SR)
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


def save_sensor_model(path, clf, filename):
    """Saves sensor model to disk"""
    with open(os.path.join(path, filename), "wb") as f:
        pickle.dump(clf, f, protocol=PICKLE_PROTOCOL)


def plot_spectra(spectra, labels, save_path=None):
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

    if save_path:
        fig.savefig(save_path)
        print(f"Spectra plot saved to {save_path}")
    fig.show()


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """Plot confusion matrix for evaluation."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=pyplot.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=numpy.arange(cm.shape[1]),
        yticks=numpy.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    pyplot.show()


def plot_frequency_spectrum(spectrum, save_path=None):
    """Plot frequency spectrum for presentations."""
    fig, ax = pyplot.subplots()
    ax.plot(spectrum.index, spectrum.values)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Frequency Spectrum of Active Sound")
    ax.grid(True)
    if save_path:
        fig.savefig(save_path)
        print(f"Frequency spectrum saved to {save_path}")
    pyplot.show()


def plot_recorded_spectra(data_dir, classes, save_path=None):
    """Plot frequency spectra of recorded samples for presentations."""
    fig, ax = pyplot.subplots()
    color_list = pyplot.rcParams["axes.prop_cycle"].by_key()["color"]

    # Cycle through colors if we have more classes than colors
    cdict = {}
    for i, cls in enumerate(classes):
        cdict[cls] = color_list[i % len(color_list)]

    for cls in classes:
        # Find first file for this class
        files = [f for f in os.listdir(data_dir) if f.startswith("1_") and cls in f]
        if files:
            file_path = os.path.join(data_dir, files[0])
            audio = preprocessing.load_audio(file_path, sr=SR)
            spectrum = preprocessing.audio_to_features(audio)
            ax.plot(spectrum.index, spectrum.values, label=cls, color=cdict[cls])

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Frequency Spectra of Recorded Samples")
    ax.legend()
    ax.grid(True)
    if save_path:
        fig.savefig(save_path)
        print(f"Recorded spectra saved to {save_path}")
    # pyplot.show()  # Removed to avoid conflicts with main pyplot.show()


def plot_waveforms(data_dir, classes, save_path=None):
    """Plot average waveforms for each class."""
    fig, axes = pyplot.subplots(
        len(classes), 1, figsize=(10, 5 * len(classes)), sharex=True
    )
    if len(classes) == 1:
        axes = [axes]

    for i, cls in enumerate(classes):
        files = [f for f in os.listdir(data_dir) if f.startswith("1_") and cls in f]
        if files:
            audios = [
                preprocessing.load_audio(os.path.join(data_dir, f), sr=SR)
                for f in files
            ]
            avg_audio = numpy.mean(audios, axis=0)  # Average across samples
            print(
                f"Plotting average waveform for class '{cls}' from {len(files)} files"
            )
            axes[i].plot(avg_audio, label=f"Class: {cls} (avg of {len(files)})")
            axes[i].set_title(f"Average Waveform for class '{cls}'")
            axes[i].set_ylabel("Amplitude")
            axes[i].legend()
        else:
            print(f"No files found for class '{cls}' in {data_dir}")

    axes[-1].set_xlabel("Time (samples)")
    pyplot.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Waveforms saved to {save_path}")
    pyplot.show()


def plot_class_spectra(data_dir, classes, save_path=None):
    """Plot frequency spectra for one example per class."""
    fig, axes = pyplot.subplots(
        len(classes), 1, figsize=(10, 5 * len(classes)), sharex=True
    )
    if len(classes) == 1:
        axes = [axes]

    for i, cls in enumerate(classes):
        files = [f for f in os.listdir(data_dir) if f.startswith("1_") and cls in f]
        if files:
            spectra = []
            for f in files:
                audio = preprocessing.load_audio(os.path.join(data_dir, f), sr=SR)
                spectrum = preprocessing.audio_to_features(audio)
                spectra.append(spectrum.values)
            avg_spectrum = numpy.mean(spectra, axis=0)
            print(
                f"Plotting average spectrum for class '{cls}' from {len(files)} files"
            )
            # Assuming spectrum.index is the same for all
            axes[i].plot(
                spectrum.index,
                avg_spectrum,
                label=f"Class: {cls} (avg of {len(files)})",
            )
            axes[i].set_title(f"Average Frequency Spectrum for class '{cls}'")
            axes[i].set_ylabel("Amplitude")
            axes[i].legend()
        else:
            print(f"No files found for class '{cls}' in {data_dir}")

    axes[-1].set_xlabel("Frequency (Hz)")
    pyplot.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Class spectra saved to {save_path}")
    pyplot.show()


def plot_spectrograms(data_dir, classes, save_path=None):
    """Plot spectrograms for one example per class."""
    fig, axes = pyplot.subplots(
        len(classes), 1, figsize=(10, 5 * len(classes)), sharex=True
    )
    if len(classes) == 1:
        axes = [axes]

    for i, cls in enumerate(classes):
        files = [f for f in os.listdir(data_dir) if f.startswith("1_") and cls in f]
        if files:
            file_path = os.path.join(data_dir, files[0])
            print(
                f"Plotting spectrogram for class '{cls}' from file: {file_path} (showing one example)"
            )
            audio = preprocessing.load_audio(file_path, sr=SR)
            D = librosa.amplitude_to_db(numpy.abs(librosa.stft(audio)), ref=numpy.max)
            img = librosa.display.specshow(
                D, x_axis="time", y_axis="log", ax=axes[i], sr=SR
            )
            axes[i].set_title(f"Spectrogram for class '{cls}'")
            fig.colorbar(img, ax=axes[i], format="%+2.0f dB")
        else:
            print(f"No files found for class '{cls}' in {data_dir}")

    pyplot.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Spectrograms saved to {save_path}")
    pyplot.show()


def main():
    print("Running for model '{}'".format(MODEL_NAME))
    global DATA_DIR
    DATA_DIR = os.path.join(f"data/{BASE_DIR}", MODEL_NAME)

    # Load config
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model_type = config["active_model"]
    params = config["models"][model_type]
    tune = config.get("tune_hyperparameters", False)
    param_grid = config.get("param_grids", {}).get(model_type, {})
    feature_method = config.get("feature_method", "stft")

    sounds, labels = load_sounds(os.path.join(DATA_DIR, "data"))
    # spectra = [sound_to_spectrum(sound) for sound in sounds]
    spectra = [
        preprocessing.audio_to_features(sound, method=feature_method)
        for sound in sounds
    ]
    classes = list(set(labels))

    if SHOW_PLOTS:
        if feature_method == "stft":
            plot_spectra(
                spectra, labels, save_path=os.path.join(DATA_DIR, "spectra_plot.png")
            )
        plot_recorded_spectra(
            os.path.join(DATA_DIR, "data"),
            classes,
            save_path=os.path.join(DATA_DIR, "recorded_spectra.png"),
        )
        plot_waveforms(
            os.path.join(DATA_DIR, "data"),
            classes,
            save_path=os.path.join(DATA_DIR, "waveforms.png"),
        )
        plot_class_spectra(
            os.path.join(DATA_DIR, "data"),
            classes,
            save_path=os.path.join(DATA_DIR, "class_spectra.png"),
        )
        plot_spectrograms(
            os.path.join(DATA_DIR, "data"),
            classes,
            save_path=os.path.join(DATA_DIR, "spectrograms.png"),
        )

    if TEST_SIZE > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            spectra, labels, test_size=TEST_SIZE, stratify=labels, random_state=42
        )
    else:
        X_train, y_train = (spectra, labels)

    # Instantiate classifier based on config
    if model_type == "knn":
        base_clf = KNeighborsClassifier(**params)
    elif model_type == "svm":
        base_clf = SVC(**params)
    else:
        raise ValueError(f"Unsupported model: {model_type}")

    if tune and param_grid:
        print(f"Tuning hyperparameters for {model_type} with grid: {param_grid}")
        clf = GridSearchCV(base_clf, param_grid, cv=5, scoring="accuracy")
        clf.fit(X_train, y_train)
        best_clf = clf.best_estimator_
        print(f"Best params: {clf.best_params_}")
    else:
        best_clf = base_clf
        best_clf.fit(X_train, y_train)

    print(f"Using model: {model_type} with final params: {best_clf.get_params()}")
    train_score = best_clf.score(X_train, y_train)
    print("Fitted sensor model to data!")
    print("Training score: {:.2f}".format(train_score))

    if TEST_SIZE > 0:
        test_score = best_clf.score(X_test, y_test)
        print("Test score: {:.2f}".format(test_score))

        y_pred = best_clf.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        if SHOW_PLOTS:
            plot_confusion_matrix(
                y_test,
                y_pred,
                sorted(classes),
                save_path=os.path.join(DATA_DIR, "confusion_matrix.png"),
            )

    save_sensor_model(DATA_DIR, best_clf, SENSORMODEL_FILENAME)
    print("\nSaved model to '{}'".format(os.path.join(DATA_DIR, SENSORMODEL_FILENAME)))

    if SHOW_PLOTS:
        pyplot.pause(0.1)
        pyplot.show()


if __name__ == "__main__":
    main()
