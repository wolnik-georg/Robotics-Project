#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script for the "Acoustic Sensing Starter Kit"
[Zöller, Gabriel, Vincent Wall, and Oliver Brock. "Active Acoustic Contact Sensing for Soft Pneumatic Actuators." In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.]

This script _records_ data samples for different classes, e.g. contact locations.

In 'USER SETTINGS' define:
BASE_DIR - path where data should be stored
SOUND_NAME - type of active sound to use. choose from SOUNDS or create your own.
CLASS_LABELS - labels of the different prediction classes, e.g. contact locations.
SAMPLES_PER_CLASS - how many samples to record per class
MODEL_NAME - name of the model. is used as folder name.
SHUFFLE_RECORDING_ORDER - whether or not to randomize the recording order

Before running the script, make sure to start QjackCtl.

@author: Vincent Wall, Gabriel Zöller
@copyright 2020 Robotics and Biology Lab, TU Berlin
@licence: BSD Licence
"""
import numpy
import random
import librosa
import os
import time
import scipy.io.wavfile
import scipy.signal
from matplotlib import pyplot  # type: ignore
from matplotlib.widgets import Button  # type: ignore
from jacktools.jacksignal import JackSignal
from glob import glob

# ==================
# USER SETTINGS
# ==================
BASE_DIR = "."
SOUND_NAME = "sweep"  # sound to use
CLASS_LABELS = ["tip", "middle", "base", "back", "none"]  # classes to train
DEMO_CLASS_LABELS = ["tap", "no_tap"]
SAMPLES_PER_CLASS = 50
MODEL_NAME = "material_tapping_demo"
SHUFFLE_RECORDING_ORDER = False
APPEND_TO_EXISTING_FILES = True
# ==================


CHANNELS = 1
SR = 48000

# Example sounds
RECORDING_DELAY_SILENCE = numpy.zeros(
    int(SR * 0.15), dtype="float32"
)  # the microphone has about .15 seconds delay in recording the sound
SOUNDS = dict(
    {
        "sweep": numpy.hstack(
            [
                scipy.signal.chirp(numpy.arange(SR) / SR, 20, 1, 20000).astype(
                    "float32"
                ),
                RECORDING_DELAY_SILENCE,
            ]
        ),
        "quick_sweep": numpy.hstack(
            [
                scipy.signal.chirp(
                    numpy.arange(int(SR * 0.3)) / SR, 20, 0.3, 20000
                ).astype("float32"),
                RECORDING_DELAY_SILENCE,
            ]
        ),
        "white_noise": numpy.hstack(
            [
                numpy.random.uniform(low=-0.999, high=1.0, size=(SR)).astype("float32"),
                RECORDING_DELAY_SILENCE,
            ]
        ),
        "silence": numpy.hstack(
            [numpy.zeros((SR,), dtype="float32"), RECORDING_DELAY_SILENCE]
        ),
    }
)


def main():
    print("Running for model '{}'".format(MODEL_NAME))
    print("Using sound: {}".format(SOUND_NAME))
    print(
        "and classes: {} (recording {} samples per class sequentially)".format(
            DEMO_CLASS_LABELS, SAMPLES_PER_CLASS
        )
    )

    # check if data was previously recorded
    # ask if want to load or re-record and overwrite
    global DATA_DIR
    DATA_DIR = mkpath(BASE_DIR, MODEL_NAME)

    setup_experiment()
    setup_jack(SOUND_NAME)
    setup_matplotlib()


def setup_experiment():
    global label_order
    global samples_remaining
    global current_label_idx
    global current_label
    global sample_id

    if SHUFFLE_RECORDING_ORDER:
        label_order = random.sample(DEMO_CLASS_LABELS, len(DEMO_CLASS_LABELS))
    else:
        label_order = DEMO_CLASS_LABELS[:]

    samples_remaining = {label: SAMPLES_PER_CLASS for label in DEMO_CLASS_LABELS}
    sample_id = {label: 0 for label in DEMO_CLASS_LABELS}
    current_label_idx = 0
    current_label = label_order[0] if label_order else None

    if APPEND_TO_EXISTING_FILES:
        existing_files = glob(DATA_DIR + "/*.wav")
        for f in existing_files:
            basename = os.path.basename(f)
            if basename.endswith(".wav"):
                parts = basename[:-4].split("_", 1)
                if len(parts) == 2:
                    id_str, label = parts
                    try:
                        id_num = int(id_str)
                        if label in samples_remaining:
                            sample_id[label] = max(sample_id[label], id_num + 1)
                            samples_remaining[label] = max(
                                0, samples_remaining[label] - 1
                            )
                    except ValueError:
                        pass


def setup_jack(sound_name):
    global J
    global Ains
    J = JackSignal("JS")
    print(J.get_state())
    assert J.get_state() >= 0, "Creating JackSignal failed."
    name, sr, period = J.get_jack_info()

    for i in range(CHANNELS):
        J.create_output(i, "out_{}".format(i))
        J.create_input(i, "in_{}".format(i))
        J.connect_input(i, "system:capture_{}".format(i + 1))
        J.connect_output(i, "system:playback_{}".format(i + 1))
    J.silence()

    sound = SOUNDS[sound_name]
    Aouts = [sound] * CHANNELS
    Ains = [numpy.zeros_like(sound, dtype=numpy.float32) for __ in range(CHANNELS)]
    for i in range(CHANNELS):
        J.set_output_data(i, Aouts[i])
        J.set_input_data(i, Ains[i])

    # store active sound for reference
    sound_file = os.path.join(DATA_DIR, "{}_{}.wav".format(0, sound_name))
    scipy.io.wavfile.write(sound_file, SR, sound)
    return J, Aouts, Ains


def setup_matplotlib():
    global LINES
    global TITLE
    global b_rec
    fig, ax = pyplot.subplots(1)
    ax.set_ylim(-1, 1)
    pyplot.subplots_adjust(bottom=0.2)
    (LINES,) = ax.plot(Ains[0])
    ax_back = pyplot.axes([0.59, 0.05, 0.1, 0.075])
    b_back = Button(ax_back, "[B]ack")
    b_back.on_clicked(back)
    ax_rec = pyplot.axes([0.81, 0.05, 0.1, 0.075])
    b_rec = Button(ax_rec, "[R]ecord")
    b_rec.on_clicked(record)
    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    TITLE = ax.set_title(get_current_title())
    pyplot.show()


def on_key(event):
    if event.key == "r":
        record(event)
    elif event.key == "b":
        back(event)


def get_current_title():
    name = "Model: {}".format(MODEL_NAME.replace("_", " "))
    prev_label = label_order[current_label_idx - 1] if current_label_idx > 0 else ""
    next_label = (
        label_order[current_label_idx + 1]
        if current_label_idx + 1 < len(label_order)
        else ""
    )
    labels = "previous: {}   current: [{}]   next: {}".format(
        prev_label, current_label or "", next_label
    )
    remaining = samples_remaining.get(current_label, 0) if current_label else 0
    number = "Remaining for {}: {}".format(current_label or "", remaining)
    if current_label_idx >= len(label_order):
        number += " DONE!"
    title = "{}\n{}\n{}".format(name, labels, number)
    return title


def back(event):
    global current_label_idx
    global current_label
    # Go back to previous label
    current_label_idx = max(0, current_label_idx - 1)
    current_label = (
        label_order[current_label_idx] if current_label_idx < len(label_order) else None
    )
    update()


def record(event):
    global current_label_idx
    global current_label
    if not label_order or current_label_idx >= len(label_order):
        print("All recordings completed.")
        return

    while samples_remaining[current_label] > 0:
        print(
            f"Recording for '{current_label}' - {'Tap on the surface' if current_label == 'tap' else 'Do NOT tap - let sound play'} and get ready..."
        )
        for i in range(1, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("NOW!")
        J.process()
        J.wait()
        LINES.set_ydata(Ains[0].reshape(-1))
        store()
        samples_remaining[current_label] -= 1
        pyplot.draw()

    # Advance to next label
    current_label_idx += 1
    if current_label_idx < len(label_order):
        current_label = label_order[current_label_idx]
    else:
        current_label = None
    update()


def store():
    global sample_id
    sample_id[current_label] += 1
    sound_file = os.path.join(
        DATA_DIR, "{}_{}.wav".format(sample_id[current_label], current_label)
    )
    scipy.io.wavfile.write(sound_file, SR, Ains[0])


def mkpath(*args):
    """Takes parts of a path (dir or file), joins them, creates the directory if it doesn't exist and returns the path.
    figure_path = mkpath(PLOT_DIR, "experiment", "figure.svg")
    """
    path = os.path.join(*args)
    if os.path.splitext(path)[1]:  # if path has file extension
        base_path = os.path.split(path)[0]
    else:
        base_path = path
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    return path


def update():
    TITLE.set_text(get_current_title())
    pyplot.draw()


if __name__ == "__main__":
    main()
