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

# Class ordering: Set "class_order" in config.json to a custom array like ["void", "contact"]
# for specific ordering. Set to null for automatic sorting (numerical for frequencies, alphabetical otherwise)

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


def save_sensor_model(path, clf, filename, classes=None):
    """Saves sensor model to disk with class information for consistent ordering"""
    model_data = {"model": clf, "classes": classes, "version": "1.0"}
    with open(os.path.join(path, filename), "wb") as f:
        pickle.dump(model_data, f, protocol=PICKLE_PROTOCOL)


def plot_spectra(spectra, labels, save_path=None):
    from matplotlib import pyplot

    fig, ax = pyplot.subplots(1)
    # Use maximally distinct colors for optimal class differentiation
    color_list = [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#FFA500",
        "#800080",
        "#FFC0CB",
        "#A52A2A",
        "#808080",
        "#000000",
        "#8B4513",
        "#800000",
        "#808000",
        "#008000",
        "#008080",
        "#000080",
        "#FF6347",
        "#32CD32",
    ]
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
    fig, ax = pyplot.subplots(figsize=(12, 10), dpi=150)

    # Create heatmap with better styling
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", alpha=0.8)

    # Add colorbar with better formatting
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Number of Samples", fontsize=12, fontweight="bold")

    # Configure axes
    ax.set(
        xticks=numpy.arange(cm.shape[1]),
        yticks=numpy.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    # Rotate x-axis labels for better readability
    pyplot.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=10,
    )
    pyplot.setp(ax.get_yticklabels(), fontsize=10)

    # Add text annotations on the confusion matrix
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )

    # Improve title and axis labels
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Predicted label", fontsize=14, fontweight="bold")
    ax.set_ylabel("True label", fontsize=14, fontweight="bold")

    fig.tight_layout()
    if save_path:
        fig.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Confusion matrix saved to {save_path}")
    # pyplot.show()  # Removed to avoid conflicts with main pyplot.show()


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
    # High-quality figure settings
    fig, ax = pyplot.subplots(figsize=(12, 8), dpi=150)  # Larger size, higher DPI

    # Use maximally distinct colors for optimal class differentiation
    distinct_colors = [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#FFA500",
        "#800080",
        "#FFC0CB",
        "#A52A2A",
        "#808080",
        "#000000",
        "#8B4513",
        "#800000",
        "#808000",
        "#008000",
        "#008080",
        "#000080",
        "#FF6347",
        "#32CD32",
    ]

    # Cycle through distinct colors if we have more classes than colors
    cdict = {}
    for i, cls in enumerate(classes):
        cdict[cls] = distinct_colors[i % len(distinct_colors)]

    for cls in classes:
        # Find first file for this class
        files = [f for f in os.listdir(data_dir) if f.startswith("1_") and cls in f]
        if files:
            file_path = os.path.join(data_dir, files[0])
            audio = preprocessing.load_audio(file_path, sr=SR)
            spectrum = preprocessing.audio_to_features(audio)
            # Thicker lines for better visibility
            ax.plot(
                spectrum.index,
                spectrum.values,
                label=cls,
                color=cdict[cls],
                linewidth=2,
                alpha=0.8,
            )

    # Improved styling
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Amplitude", fontsize=14, fontweight="bold")
    ax.set_title(
        "Frequency Spectra of Recorded Samples", fontsize=16, fontweight="bold", pad=20
    )

    # Better legend
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Improved grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Better axis styling
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pyplot.tight_layout()

    if save_path:
        fig.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"High-quality PNG saved to {save_path}")

    # pyplot.show()  # Removed to avoid conflicts with main pyplot.show()


def plot_waveforms(data_dir, classes, save_path=None):
    """Plot average waveforms for each class in separate subplots."""
    fig, axes = pyplot.subplots(
        len(classes), 1, figsize=(14, 6 * len(classes)), sharex=False, dpi=150
    )
    if len(classes) == 1:
        axes = [axes]

    # Use maximally distinct colors for optimal class differentiation
    color_list = [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#FFA500",
        "#800080",
        "#FFC0CB",
        "#A52A2A",
        "#808080",
        "#000000",
        "#FF1493",
        "#800000",
        "#808000",
        "#008000",
        "#008080",
        "#000080",
        "#FF6347",
        "#32CD32",
    ]

    # Cycle through colors if we have more classes than colors
    cdict = {}
    for i, cls in enumerate(classes):
        cdict[cls] = color_list[i % len(color_list)]

    for i, cls in enumerate(classes):
        ax = axes[i]
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
            # Plot waveform for this class
            time_axis = numpy.arange(len(avg_audio)) / SR  # Convert samples to seconds
            ax.plot(
                time_axis,
                avg_audio,
                linewidth=2,
                alpha=0.8,
                color=cdict[cls],
            )
            ax.set_ylabel("Amplitude", fontsize=12, fontweight="bold")
            ax.set_title(f"Class: {cls}", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel("Time (seconds)", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
        else:
            print(f"No files found for class '{cls}' in {data_dir}")
            ax.set_ylabel("Amplitude", fontsize=12, fontweight="bold")
            ax.set_title(f"Class: {cls}", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel("Time (seconds)", fontsize=14, fontweight="bold")

    # Overall title
    fig.suptitle("Waveform per Class", fontsize=16, fontweight="bold", y=0.93)

    # Adjust spacing to ensure x-axis label is clearly visible
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.3, hspace=0.4)

    if save_path:
        fig.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Waveforms saved to {save_path}")

    # pyplot.show()  # Removed to avoid conflicts with main pyplot.show()


def plot_class_spectra(data_dir, classes, save_path=None):
    """Plot frequency spectra for one example per class."""
    fig, axes = pyplot.subplots(
        len(classes), 1, figsize=(14, 6 * len(classes)), sharex=True, dpi=150
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
                linewidth=2,
                alpha=0.8,
            )
            axes[i].set_title(
                f"Average Frequency Spectrum for class '{cls}'",
                fontsize=14,
                fontweight="bold",
            )
            axes[i].set_ylabel("Amplitude", fontsize=12, fontweight="bold")
            axes[i].legend(fontsize=11)
            axes[i].grid(True, alpha=0.3, linestyle="--")
            axes[i].tick_params(axis="both", which="major", labelsize=11)
        else:
            print(f"No files found for class '{cls}' in {data_dir}")

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=12, fontweight="bold")
    pyplot.tight_layout()
    if save_path:
        fig.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Class spectra saved to {save_path}")
    # pyplot.show()  # Removed to avoid conflicts with main pyplot.show()


def plot_spectrograms(data_dir, classes, save_path=None):
    """Plot spectrograms for one example per class."""
    fig, axes = pyplot.subplots(
        len(classes), 1, figsize=(14, 6 * len(classes)), sharex=True, dpi=150
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
                D, x_axis="time", y_axis="log", ax=axes[i], sr=SR, cmap="viridis"
            )
            axes[i].set_title(
                f"Spectrogram for class '{cls}'", fontsize=14, fontweight="bold"
            )
            # Add colorbar with better formatting
            cbar = fig.colorbar(
                img, ax=axes[i], format="%+2.0f dB", shrink=0.8, aspect=20
            )
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Amplitude (dB)", fontsize=11, fontweight="bold")
        else:
            print(f"No files found for class '{cls}' in {data_dir}")

    pyplot.tight_layout()
    if save_path:
        fig.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Spectrograms saved to {save_path}")
    # pyplot.show()  # Removed to avoid conflicts with main pyplot.show()


def main():
    print("Running for model '{}'".format(MODEL_NAME))
    global DATA_DIR
    DATA_DIR = os.path.join(BASE_DIR, "data", MODEL_NAME)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "../configs/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    model_type = config["active_model"]
    params = config["models"][model_type]
    tune = config.get("tune_hyperparameters", False)
    param_grid = config.get("param_grids", {}).get(model_type, {})
    feature_method = config.get("feature_method", "stft")
    custom_class_order = config.get("class_order", None)

    sounds, labels = load_sounds(os.path.join(DATA_DIR, "data"))
    # spectra = [sound_to_spectrum(sound) for sound in sounds]
    spectra = [
        preprocessing.audio_to_features(sound, method=feature_method)
        for sound in sounds
    ]

    # Ensure consistent class ordering for reproducible results
    unique_labels = list(set(labels))

    if custom_class_order:
        # Use custom order specified in config
        classes = custom_class_order
        # Validate that all classes are present
        missing_classes = set(unique_labels) - set(custom_class_order)
        extra_classes = set(custom_class_order) - set(unique_labels)
        if missing_classes:
            print(f"Warning: Custom class order missing classes: {missing_classes}")
        if extra_classes:
            print(f"Warning: Custom class order has extra classes: {extra_classes}")
    else:
        # Sort classes: try numerical sorting for frequency-based labels, otherwise alphabetical
        try:
            # Try to sort numerically (works for "100 Hz", "200 Hz", etc.)
            classes = sorted(
                unique_labels,
                key=lambda x: float(x.split()[0]) if x.split()[0].isdigit() else x,
            )
        except (ValueError, IndexError):
            # Fall back to alphabetical sorting
            classes = sorted(unique_labels)

    print(f"Classes (in consistent order): {classes}")

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
                classes,  # Use the consistently ordered classes
                save_path=os.path.join(DATA_DIR, "confusion_matrix.png"),
            )

    save_sensor_model(DATA_DIR, best_clf, SENSORMODEL_FILENAME, classes)
    print("\nSaved model to '{}'".format(os.path.join(DATA_DIR, SENSORMODEL_FILENAME)))

    if SHOW_PLOTS:
        pyplot.pause(0.1)
        pyplot.show()


if __name__ == "__main__":
    main()
