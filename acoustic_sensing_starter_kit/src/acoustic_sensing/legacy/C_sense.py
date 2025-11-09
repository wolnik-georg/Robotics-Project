#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script for the "Acoustic Sensing Starter Kit"
[Zöller, Gabriel, Vincent Wall, and Oliver Brock. “Active Acoustic Contact Sensing for Soft Pneumatic Actuators.” In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.]

This script performs _live_sensing_ using the previously trained sensor model.

In 'USER SETTINGS' define:
BASE_DIR - path where data is read from
MODEL_NAME - name of the sensor model. used as folder name.
SENSORMODEL_FILENAME - name of the sensor model saved in the previous step.
CONTINUOUSLY - chose between continuous sensing or manually triggered

@author: Vincent Wall, Gabriel Zöller
@copyright 2020 Robotics and Biology Lab, TU Berlin
@licence: BSD Licence
"""

import numpy
import librosa
import os
import sys
import pandas
import pickle
import json

from sklearn.neighbors import KNeighborsClassifier
from jacktools.jacksignal import JackSignal
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from A_record import MODEL_NAME
from B_train import SENSORMODEL_FILENAME
import preprocessing

# ==================
# USER SETTINGS
# ==================
BASE_DIR = "."
CONTINUOUSLY = True  # chose between continuous sensing or manually triggered
LOG_DATA = True  # enable data logging for reconstruction
LOG_FILE = "sensing_log.csv"  # file to save sensing data
# ==================

CHANNELS = 1
SR = 48000

DATA_DIR = None
predictor = None

is_paused = False

# Python 2/3 compatibility - raw_input was renamed to input in Python 3
if sys.version_info.major == 2:
    input = raw_input


class LiveAcousticSensor(object):
    def __init__(self):
        # load sound from file (starts with "0_")
        active_sound_filename = [
            fn for fn in os.listdir(os.path.join(DATA_DIR, "data")) if fn[:2] == "0_"
        ][0]
        self.sound = (
            preprocessing.load_audio(
                os.path.join(DATA_DIR, "data", active_sound_filename), sr=SR
            )
            .reshape(-1)
            .astype(numpy.float32)
        )
        self.setup_jack()
        self.setup_model()
        self.setup_window()

    def setup_jack(self):
        self.J = JackSignal("JS")
        assert self.J.get_state() >= 0, "Creating JackSignal failed."
        name, sr, period = self.J.get_jack_info()

        for i in range(CHANNELS):
            self.J.create_output(i, "out_{}".format(i))
            self.J.create_input(i, "in_{}".format(i))
            self.J.connect_input(i, "system:capture_{}".format(i + 1))
            self.J.connect_output(i, "system:playback_{}".format(i + 1))
        self.J.silence()

        self.Aouts = [self.sound] * CHANNELS
        self.Ains = [
            numpy.zeros_like(self.sound, dtype=numpy.float32) for __ in range(CHANNELS)
        ]
        for i in range(CHANNELS):
            self.J.set_output_data(i, self.Aouts[i])
            self.J.set_input_data(i, self.Ains[i])

    def setup_model(self):
        model_path = os.path.join(DATA_DIR, SENSORMODEL_FILENAME)
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Handle both old format (just model) and new format (model + classes)
        if isinstance(model_data, dict) and "model" in model_data:
            self.clf = model_data["model"]
            if "classes" in model_data and model_data["classes"] is not None:
                # Ensure classes are in the expected order
                self.model_classes = model_data["classes"]
                print(f"Model classes (consistent order): {self.model_classes}")
            else:
                self.model_classes = list(self.clf.classes_)
        else:
            # Backward compatibility with old model format
            self.clf = model_data
            self.model_classes = list(self.clf.classes_)

        print(f"Classifier classes: {self.clf.classes_}")

        # Load feature method from config
        config_path = os.path.join(os.path.dirname(__file__), "../configs/config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        self.feature_method = config.get("feature_method", "stft")

    def setup_window(self):
        f = plt.figure(1, figsize=(16, 10))
        f.clear()
        f.suptitle(
            "Acoustic Sensing for Geometric Reconstruction",
            size=16,
            fontweight="bold",
        )

        ax1 = f.add_subplot(2, 3, 1)
        ax1.set_title("Input Waveform", size=14, fontweight="bold")
        ax1.set_xlabel("Time [samples]", fontsize=10)
        ax1.set_ylabel("Amplitude", fontsize=10)
        ax1.set_ylim([-1, 1])
        ax1.grid(True, alpha=0.3)

        ax2 = f.add_subplot(2, 3, 2)
        ax2.set_title("Frequency Spectrum (STFT)", size=14, fontweight="bold")
        ax2.set_xlabel("Frequency [Hz]", fontsize=10)
        ax2.set_ylabel("Magnitude", fontsize=10)
        ax2.set_ylim([0, 250])
        ax2.grid(True, alpha=0.3)

        # Initialize plot lines for real-time updates
        # Initialize with dummy data - will be updated in predict()
        (self.wavelines,) = ax1.plot(
            numpy.arange(48000), numpy.zeros(48000), "b-", linewidth=1
        )
        # For spectrum, use frequency range 0-24000 Hz (Nyquist frequency at 48kHz)
        freq_bins = numpy.linspace(0, 24000, 2049)  # 2049 is typical for n_fft=4096
        (self.spectrumlines,) = ax2.plot(
            freq_bins, numpy.zeros(len(freq_bins)), "r-", linewidth=1
        )

        # Audio level indicator
        ax3 = f.add_subplot(2, 3, 3)
        ax3.set_title("Audio Level", size=14, fontweight="bold")
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        self.level_bar = ax3.barh([0.5], [0], height=0.8, color="#1f77b4", alpha=0.7)
        ax3.text(
            0.5, 0.5, "0%", ha="center", va="center", fontsize=12, fontweight="bold"
        )

        # Classification results (bottom row - spans 3 columns properly)
        ax4 = f.add_subplot(2, 3, (4, 6))
        ax4.set_title("Classification Results", size=14, fontweight="bold")
        ax4.axis("off")
        ax4.text(0.02, 0.8, "Current Prediction:", fontsize=16, fontweight="bold")
        self.predictiontext = ax4.text(
            0.02,
            0.6,
            "Waiting...",
            fontsize=24,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2ca02c", alpha=0.1),
        )

        # Status and keyboard shortcuts
        ax4.text(
            0.02,
            0.3,
            "Status: Running",
            fontsize=12,
            fontweight="bold",
            color="green",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.3),
        )
        self.status_text = ax4.text(
            0.02, 0.3, "Status: Running", fontsize=12, fontweight="bold", color="green"
        )

        ax4.text(
            0.02,
            0.1,
            f"Data Logging: {'Enabled' if LOG_DATA else 'Disabled'} • Controls: [P]ause • [Q]uit",
            fontsize=9,
            style="italic",
            color="blue" if LOG_DATA else "gray",
        )

        # Pause button
        ax_pause = plt.axes([0.85, 0.02, 0.12, 0.06])
        self.b_pause = Button(ax_pause, "[P]ause", color="lightgray", hovercolor="gray")
        self.b_pause.on_clicked(toggle_pause)

        # Connect keyboard events
        cid = f.canvas.mpl_connect("key_press_event", on_key)

        # Use manual layout instead of tight_layout to avoid compatibility issues
        f.subplots_adjust(
            left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4
        )
        f.show()
        plt.draw()
        plt.pause(0.00001)

    def predict(self):
        for i in range(CHANNELS):
            # spectrum = self.sound_to_spectrum(self.Ains[i])
            spectrum = preprocessing.audio_to_features(
                self.Ains[i], method=self.feature_method
            )
            prediction = self.clf.predict([spectrum])

            # Get prediction confidence if available
            try:
                probabilities = self.clf.predict_proba([spectrum])[0]
                confidence = max(probabilities) * 100
                prediction_text = f"{prediction[0]}\n({confidence:.1f}% confidence)"
            except:
                # Fallback if predict_proba not available
                prediction_text = prediction[0]

        # Use STFT for visualization (more interpretable than combined features)
        display_spectrum = preprocessing.audio_to_features(self.Ains[0], method="stft")

        # Update audio level indicator
        audio_level = numpy.sqrt(numpy.mean(self.Ains[0] ** 2))  # RMS level
        normalized_level = min(audio_level * 10, 1.0)  # Scale for visualization
        self.level_bar[0].set_width(normalized_level)

        # Update level text
        level_text = f"{normalized_level*100:.1f}%"
        # Clear previous text and add new
        for txt in self.level_bar[0].axes.texts:
            txt.remove()
        self.level_bar[0].axes.text(
            0.5,
            0.5,
            level_text,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        self.wavelines.set_data(
            numpy.arange(len(self.Ains[0])), self.Ains[0].reshape(-1)
        )
        self.spectrumlines.set_data(display_spectrum.index, display_spectrum.values)

        self.predictiontext.set_text(prediction_text)

        # Log data for reconstruction if enabled
        if LOG_DATA:
            self.log_sensing_data(prediction_text, confidence)

        plt.draw()
        plt.pause(0.00001)

    def log_sensing_data(self, prediction, confidence):
        """Log sensing data for geometric reconstruction analysis"""
        import time
        import csv
        import os

        # Create log entry
        timestamp = time.time()
        log_entry = {
            "timestamp": timestamp,
            "prediction": (
                prediction.split("\n")[0] if "\n" in prediction else prediction
            ),  # Extract just the frequency
            "confidence": confidence if confidence is not None else 0.0,
            "audio_level": numpy.sqrt(numpy.mean(self.Ains[0] ** 2)),
            "feature_method": self.feature_method,
        }

        # Write to CSV
        log_path = os.path.join(DATA_DIR, LOG_FILE)
        file_exists = os.path.isfile(log_path)

        with open(log_path, "a", newline="") as csvfile:
            fieldnames = [
                "timestamp",
                "prediction",
                "confidence",
                "audio_level",
                "feature_method",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(log_entry)

    def run(self):
        if CONTINUOUSLY:
            while True:
                if not is_paused:
                    self.J.process()
                    self.J.wait()
                    self.predict()
                plt.pause(1)
        else:
            key = input("Press <Enter> to sense! ('q' to abort)")
            while key == "":
                self.J.process()
                self.J.wait()
                self.predict()
                key = input("Press <Enter> to sense! ('q' to abort)")


def toggle_pause(event):
    global is_paused
    is_paused = not is_paused
    # Update button text based on pause state
    if hasattr(predictor, "b_pause"):
        if is_paused:
            predictor.b_pause.label.set_text("[R]esume")
            predictor.status_text.set_text("Status: Paused")
            predictor.status_text.set_color("orange")
        else:
            predictor.b_pause.label.set_text("[P]ause")
            predictor.status_text.set_text("Status: Running")
            predictor.status_text.set_color("green")
        plt.draw()


def on_key(event):
    if event.key == "p":
        toggle_pause(event)
    elif event.key == "q":
        sys.exit()


def main():
    global DATA_DIR
    DATA_DIR = os.path.join(BASE_DIR, "data", MODEL_NAME)
    global predictor
    predictor = LiveAcousticSensor()
    predictor.run()


if __name__ == "__main__":
    main()
