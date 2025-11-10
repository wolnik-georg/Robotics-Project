#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script for the "Acoustic Sensing Starter Kit"
[Zöller, Gabriel, Vincent Wall, and Oliver Brock. "Active Acoustic Contact Sensing for Soft Pneumatic Actuators." In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.]

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
import sys
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot

# Fix imports - add proper path resolution
script_dir = Path(__file__).parent
root_dir = script_dir.parent.parent.parent  # Go up to acoustic_sensing_starter_kit/
sys.path.append(str(root_dir / "src"))


# Create a preprocessing module inline since the original import is broken
class preprocessing:
    @staticmethod
    def load_audio(filepath, sr=48000):
        """Load audio file"""
        try:
            audio, _ = librosa.load(filepath, sr=sr)
            return audio
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    @staticmethod
    def audio_to_features(audio, method="stft"):
        """Convert audio to features"""
        if method == "stft":
            spectrum = numpy.fft.rfft(audio)
            amplitude_spectrum = numpy.abs(spectrum)
            d = 1.0 / 48000  # Using global SR
            freqs = numpy.fft.rfftfreq(len(audio), d)
            index = pandas.Index(freqs)
            series = pandas.Series(amplitude_spectrum, index=index)
            return series
        else:
            # For other methods, return the audio as-is
            return pandas.Series(audio)


# ==================
# USER SETTINGS
# ==================
BASE_DIR = root_dir  # Use the resolved root directory

try:
    from acoustic_sensing.legacy.A_record import MODEL_NAME

    print(f"Using MODEL_NAME from A_record.py: {MODEL_NAME}")
except ImportError:
    MODEL_NAME = "soft_finger_batch_4"  # fallback if A_record.py doesn't exist
    print(f"Using fallback MODEL_NAME: {MODEL_NAME}")

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
        if not fn.endswith(".wav"):
            continue

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
            if sound is not None:
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
    fig, ax = pyplot.subplots(1, figsize=(12, 8), dpi=150)
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
        ax.plot(s, c=cdict[l], alpha=0.7, linewidth=2)

    from matplotlib.lines import Line2D

    legend_lines = [Line2D([0], [0], color=col, lw=4) for col in cdict.values()]
    legend_labels = list(cdict.keys())
    ax.legend(legend_lines, legend_labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xlabel("Frequency (Hz)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Amplitude", fontsize=12, fontweight="bold")
    ax.set_title("Frequency Spectra of All Samples", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Spectra plot saved to {save_path}")
    pyplot.tight_layout()
    pyplot.show()


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

    pyplot.tight_layout()
    if save_path:
        fig.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Confusion matrix saved to {save_path}")
    pyplot.show()


def plot_frequency_spectrum(spectrum, save_path=None):
    """Plot frequency spectrum for presentations."""
    fig, ax = pyplot.subplots(figsize=(12, 8), dpi=150)
    ax.plot(spectrum.index, spectrum.values, linewidth=2, color="#1f77b4")
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Amplitude", fontsize=14, fontweight="bold")
    ax.set_title("Frequency Spectrum of Active Sound", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=12)

    pyplot.tight_layout()
    if save_path:
        fig.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Frequency spectrum saved to {save_path}")
    pyplot.show()


def plot_recorded_spectra(data_dir, classes, save_path=None):
    """Plot frequency spectra of recorded samples for presentations."""
    fig, ax = pyplot.subplots(figsize=(12, 8), dpi=150)

    # Use distinct colors
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

    cdict = {}
    for i, cls in enumerate(classes):
        cdict[cls] = distinct_colors[i % len(distinct_colors)]

    for cls in classes:
        # Find first file for this class
        files = [f for f in os.listdir(data_dir) if f.endswith(".wav") and cls in f]
        if files:
            file_path = os.path.join(data_dir, files[0])
            audio = preprocessing.load_audio(file_path, sr=SR)
            if audio is not None:
                spectrum = preprocessing.audio_to_features(audio)
                ax.plot(
                    spectrum.index,
                    spectrum.values,
                    label=cls,
                    color=cdict[cls],
                    linewidth=2,
                    alpha=0.8,
                )

    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Amplitude", fontsize=14, fontweight="bold")
    ax.set_title(
        "Frequency Spectra of Recorded Samples", fontsize=16, fontweight="bold", pad=20
    )

    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pyplot.tight_layout()
    if save_path:
        fig.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Recorded spectra saved to {save_path}")
    pyplot.show()


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
        files = [f for f in os.listdir(data_dir) if f.endswith(".wav") and cls in f]
        if files:
            audios = [
                preprocessing.load_audio(os.path.join(data_dir, f), sr=SR)
                for f in files
            ]
            # Filter out None values
            audios = [a for a in audios if a is not None]
            if audios:
                avg_audio = numpy.mean(audios, axis=0)  # Average across samples
                print(
                    f"Plotting average waveform for class '{cls}' from {len(files)} files"
                )
                # Plot waveform for this class
                time_axis = (
                    numpy.arange(len(avg_audio)) / SR
                )  # Convert samples to seconds
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

    pyplot.show()


def plot_class_spectra(data_dir, classes, save_path=None):
    """Plot frequency spectra for one example per class."""
    fig, axes = pyplot.subplots(
        len(classes), 1, figsize=(14, 6 * len(classes)), sharex=True, dpi=150
    )
    if len(classes) == 1:
        axes = [axes]

    for i, cls in enumerate(classes):
        files = [f for f in os.listdir(data_dir) if f.endswith(".wav") and cls in f]
        if files:
            spectra = []
            for f in files:
                audio = preprocessing.load_audio(os.path.join(data_dir, f), sr=SR)
                if audio is not None:
                    spectrum = preprocessing.audio_to_features(audio)
                    spectra.append(spectrum.values)
            if spectra:
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
    pyplot.show()


def plot_spectrograms(data_dir, classes, save_path=None):
    """Plot spectrograms for one example per class."""
    fig, axes = pyplot.subplots(
        len(classes), 1, figsize=(14, 6 * len(classes)), sharex=True, dpi=150
    )
    if len(classes) == 1:
        axes = [axes]

    for i, cls in enumerate(classes):
        files = [f for f in os.listdir(data_dir) if f.endswith(".wav") and cls in f]
        if files:
            file_path = os.path.join(data_dir, files[0])
            print(f"Plotting spectrogram for class '{cls}' from file: {files[0]}")
            audio = preprocessing.load_audio(file_path, sr=SR)
            if audio is not None:
                D = librosa.amplitude_to_db(
                    numpy.abs(librosa.stft(audio)), ref=numpy.max
                )
                img = librosa.display.specshow(
                    D, x_axis="time", y_axis="log", ax=axes[i], sr=SR, cmap="viridis"
                )
                axes[i].set_title(
                    f"Spectrogram for class '{cls}'", fontsize=14, fontweight="bold"
                )
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
    pyplot.show()


def main():
    print("Running for model '{}'".format(MODEL_NAME))
    global DATA_DIR

    # Find the correct data directory structure
    model_dir = BASE_DIR / "data" / MODEL_NAME

    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        print("Available model directories:")
        data_base_dir = BASE_DIR / "data"
        if data_base_dir.exists():
            for p in data_base_dir.glob("*"):
                if p.is_dir():
                    print(f"  {p.name}")
        return

    # Check for nested data directory or direct WAV files
    nested_data_dir = model_dir / "data"
    if nested_data_dir.exists() and any(nested_data_dir.glob("*.wav")):
        DATA_DIR = str(nested_data_dir)
    elif any(model_dir.glob("*.wav")):
        DATA_DIR = str(model_dir)
    else:
        print(f"No WAV files found in {model_dir} or {nested_data_dir}")
        return

    print(f"Using data directory: {DATA_DIR}")

    # Load config (create default if not found)
    config_path = BASE_DIR / "src" / "acoustic_sensing" / "configs" / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            "active_model": "knn",
            "models": {"knn": {"n_neighbors": 5}, "svm": {"C": 1.0, "kernel": "rbf"}},
            "tune_hyperparameters": False,
            "param_grids": {
                "knn": {"n_neighbors": [3, 5, 7, 9]},
                "svm": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
            },
            "feature_method": "stft",
            "class_order": None,
        }

    model_type = config["active_model"]
    params = config["models"][model_type]
    tune = config.get("tune_hyperparameters", False)
    param_grid = config.get("param_grids", {}).get(model_type, {})
    feature_method = config.get("feature_method", "stft")
    custom_class_order = config.get("class_order", None)

    sounds, labels = load_sounds(DATA_DIR)
    if len(sounds) == 0:
        print("No sounds loaded!")
        return

    # Convert to features
    print("Converting sounds to features...")
    spectra = [
        preprocessing.audio_to_features(sound, method=feature_method)
        for sound in sounds
    ]

    # Ensure consistent class ordering
    unique_labels = list(set(labels))

    if custom_class_order:
        classes = custom_class_order
        missing_classes = set(unique_labels) - set(custom_class_order)
        extra_classes = set(custom_class_order) - set(unique_labels)
        if missing_classes:
            print(f"Warning: Custom class order missing classes: {missing_classes}")
        if extra_classes:
            print(f"Warning: Custom class order has extra classes: {extra_classes}")
    else:
        try:
            classes = sorted(
                unique_labels,
                key=lambda x: float(x.split()[0]) if x.split()[0].isdigit() else x,
            )
        except (ValueError, IndexError):
            classes = sorted(unique_labels)

    print(f"Classes (in consistent order): {classes}")

    # Create visualizations
    if SHOW_PLOTS:
        print("Creating visualizations...")

        if feature_method == "stft":
            plot_spectra(
                spectra, labels, save_path=os.path.join(DATA_DIR, "spectra_plot.png")
            )

        plot_recorded_spectra(
            DATA_DIR,
            classes,
            save_path=os.path.join(DATA_DIR, "recorded_spectra.png"),
        )

        plot_waveforms(
            DATA_DIR,
            classes,
            save_path=os.path.join(DATA_DIR, "waveforms.png"),
        )

        plot_class_spectra(
            DATA_DIR,
            classes,
            save_path=os.path.join(DATA_DIR, "class_spectra.png"),
        )

        plot_spectrograms(
            DATA_DIR,
            classes,
            save_path=os.path.join(DATA_DIR, "spectrograms.png"),
        )

        # Plot frequency spectrum if we have the active sound
        if SOUND_NAME:
            active_sound_file = os.path.join(DATA_DIR, f"0_{SOUND_NAME}.wav")
            if os.path.exists(active_sound_file):
                active_audio = preprocessing.load_audio(active_sound_file, sr=SR)
                if active_audio is not None:
                    active_spectrum = preprocessing.audio_to_features(active_audio)
                    plot_frequency_spectrum(
                        active_spectrum,
                        save_path=os.path.join(DATA_DIR, "frequency_spectrum.png"),
                    )

    # Train/test split
    if TEST_SIZE > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            spectra, labels, test_size=TEST_SIZE, stratify=labels, random_state=42
        )
    else:
        X_train, y_train = (spectra, labels)

    # Instantiate classifier
    if model_type == "knn":
        base_clf = KNeighborsClassifier(**params)
    elif model_type == "svm":
        base_clf = SVC(**params)
    else:
        raise ValueError(f"Unsupported model: {model_type}")

    # Hyperparameter tuning
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

    # Evaluation
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
                classes,
                save_path=os.path.join(DATA_DIR, "confusion_matrix.png"),
            )

    # Save model
    save_sensor_model(DATA_DIR, best_clf, SENSORMODEL_FILENAME, classes)
    print("\nSaved model to '{}'".format(os.path.join(DATA_DIR, SENSORMODEL_FILENAME)))

    if SHOW_PLOTS:
        pyplot.show()


if __name__ == "__main__":
    main()
