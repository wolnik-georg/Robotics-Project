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

from sklearn.neighbors import KNeighborsClassifier
from jacktools.jacksignal import JackSignal
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from A_record import MODEL_NAME
from B_train import SENSORMODEL_FILENAME
from B_train import sound_to_spectrum, sound_to_spectrum_stft

# ==================
# USER SETTINGS
# ==================
BASE_DIR = "."
CONTINUOUSLY = True  # chose between continuous sensing or manually triggered
# ==================

CHANNELS = 1
SR = 48000

is_paused = False

plt.ion()
# plt.xkcd()

if sys.version_info.major == 2:
    input = raw_input


class LiveAcousticSensor(object):
    def __init__(self):
        # load sound from file (starts with "0_")
        active_sound_filename = [fn for fn in os.listdir(DATA_DIR) if fn[:2] == "0_"][0]
        self.sound = (
            librosa.load(os.path.join(DATA_DIR, active_sound_filename), sr=SR)[0]
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
            self.clf = pickle.load(f)
        print(self.clf.classes_)

    def setup_window(self):
        f = plt.figure(1)
        f.clear()
        f.suptitle("Acoustic Contact Sensing", size=30)
        ax1 = f.add_subplot(2, 2, 1)
        ax1.set_title("Recorded sound (waveform)", size=20)
        ax1.set_xlabel("Time [samples]")
        ax1.set_ylim([-1, 1])

        ax2 = f.add_subplot(2, 2, 2)
        ax2.set_title("Amplitude spectrum", size=20)
        ax2.set_xlabel("Frequency [Hz]")
        (self.wavelines,) = ax1.plot(self.Ains[0])
        (self.spectrumlines,) = ax2.plot(sound_to_spectrum_stft(self.Ains[0]))
        ax2.set_ylim([0, 250])

        ax3 = f.add_subplot(2, 1, 2)
        ax3.text(0.0, 0.8, "Sensing result:", dict(size=40))
        self.predictiontext = ax3.text(0.25, 0.25, "", dict(size=70))
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        # ax3.set_title("Contact location")
        ax3.axis("off")

        ax_pause = plt.axes([0.91, 0.025, 0.05, 0.075])
        self.b_pause = Button(ax_pause, "[P]ause")
        self.b_pause.on_clicked(toggle_pause)
        cid = f.canvas.mpl_connect("key_press_event", on_key)

        f.show()
        plt.draw()
        plt.pause(0.00001)

    def predict(self):
        for i in range(CHANNELS):
            # spectrum = self.sound_to_spectrum(self.Ains[i])
            spectrum = sound_to_spectrum_stft(self.Ains[i])
            prediction = self.clf.predict([spectrum])
        self.wavelines.set_ydata(self.Ains[0].reshape(-1))
        self.spectrumlines.set_ydata(spectrum)

        self.predictiontext.set_text(prediction[0])

        plt.draw()
        plt.pause(0.00001)

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


def on_key(event):
    if event.key == "p":
        toggle_pause(event)
    elif event.key == "q":
        sys.exit()


def main():
    global DATA_DIR
    DATA_DIR = os.path.join(BASE_DIR, MODEL_NAME)
    global predictor
    predictor = LiveAcousticSensor()
    predictor.run()


if __name__ == "__main__":
    main()
