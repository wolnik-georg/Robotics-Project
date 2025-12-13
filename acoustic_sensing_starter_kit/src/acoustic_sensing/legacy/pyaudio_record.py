#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyAudio version of the Acoustic Sensing Starter Kit recording script.
Adapted from A_record.py to use PyAudio instead of jacktools.

This script _records_ data samples for different classes, e.g. contact locations.

In 'USER SETTINGS' define:
BASE_DIR - path where data should be stored
SOUND_NAME - type of active sound to use. choose from SOUNDS or create your own.
CLASS_LABELS - labels of the different prediction classes, e.g. contact locations.
SAMPLES_PER_CLASS - how many samples to record per class
MODEL_NAME - name of the model. is used as folder name.
SHUFFLE_RECORDING_ORDER - whether or not to randomize the recording order

Before running the script, make sure your audio device is configured.

@author: Adapted from Vincent Wall, Gabriel Zöller
@copyright 2020 Robotics and Biology Lab, TU Berlin
@licence: BSD Licence
"""
import numpy
import random
import os
import time
import wave
import struct
import scipy.signal
import pyaudio
from matplotlib import pyplot  # type: ignore
from matplotlib.widgets import Button  # type: ignore
from glob import glob

# ==================
# USER SETTINGS
# ==================
BASE_DIR = "../../../data/"  # Save to main data folder instead of legacy folder
SOUND_NAME = "long_sweep"  # sound to use
CLASS_LABELS = ["tip", "middle", "base", "back", "none"]  # classes to train
# DEMO_CLASS_LABELS = ["finger tip", "finger middle", "finger bottom", "finger blank"]
DEMO_CLASS_LABELS = ["contact", "no_contact", "edge"]  # active classes used
SAMPLES_PER_CLASS = 10  # samples per class
MODEL_NAME = "test_finger_pyaudio"  # output folder
SHUFFLE_RECORDING_ORDER = False
APPEND_TO_EXISTING_FILES = True
CHANNELS = 1
SR = 48000
# PyAudio settings
CHUNK_SIZE = 1024
FORMAT = pyaudio.paFloat32
# ==================


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
        "long_sweep": numpy.hstack(
            [
                scipy.signal.chirp(numpy.arange(int(SR * 2)) / SR, 20, 2, 20000).astype(
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
    DATA_DIR = mkpath(BASE_DIR, MODEL_NAME, "data")

    setup_experiment()
    setup_pyaudio(SOUND_NAME)
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
        existing_files = glob(os.path.join(DATA_DIR, "*.wav"))
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


def setup_pyaudio(sound_name):
    global P
    global STREAM
    global SOUND_DATA
    global RECORDING_BUFFER

    P = pyaudio.PyAudio()

    # Get default input/output device info
    input_device_index = P.get_default_input_device_info()["index"]
    output_device_index = P.get_default_output_device_info()["index"]

    SOUND_DATA = SOUNDS[sound_name]
    RECORDING_BUFFER = numpy.zeros_like(SOUND_DATA, dtype=numpy.float32)

    # Open duplex stream for simultaneous playback and recording
    STREAM = P.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SR,
        input=True,
        output=True,
        input_device_index=input_device_index,
        output_device_index=output_device_index,
        frames_per_buffer=CHUNK_SIZE,
    )

    # Store active sound for reference
    sound_file = os.path.join(DATA_DIR, "{}_{}.wav".format(0, sound_name))
    os.makedirs(os.path.dirname(sound_file), exist_ok=True)
    # Save sound as WAV
    sound_int16 = (SOUND_DATA * 32767).astype(numpy.int16)
    with wave.open(sound_file, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SR)
        wav_file.writeframes(sound_int16.tobytes())


def setup_matplotlib():
    global LINES
    global TITLE
    global b_rec
    fig, ax = pyplot.subplots(1)
    ax.set_ylim(-1, 1)
    pyplot.subplots_adjust(bottom=0.2)
    (LINES,) = ax.plot(RECORDING_BUFFER)
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

        # Play sound and record simultaneously
        play_and_record()

        LINES.set_ydata(RECORDING_BUFFER.reshape(-1))
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


def play_and_record():
    """Play sound through output and record from input simultaneously."""
    global RECORDING_BUFFER

    # Convert sound data to bytes for PyAudio
    sound_bytes = SOUND_DATA.tobytes()
    recording_frames = []

    # Calculate number of chunks
    num_chunks = len(SOUND_DATA) // CHUNK_SIZE
    if len(SOUND_DATA) % CHUNK_SIZE != 0:
        num_chunks += 1

    for i in range(num_chunks):
        # Prepare output chunk
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, len(SOUND_DATA))
        output_chunk = SOUND_DATA[start_idx:end_idx]
        if len(output_chunk) < CHUNK_SIZE:
            # Pad with zeros if necessary
            output_chunk = numpy.pad(
                output_chunk, (0, CHUNK_SIZE - len(output_chunk)), "constant"
            )
        output_bytes = output_chunk.tobytes()

        # Write output and read input simultaneously
        STREAM.write(output_bytes)
        input_bytes = STREAM.read(CHUNK_SIZE)
        input_chunk = numpy.frombuffer(input_bytes, dtype=numpy.float32)
        recording_frames.append(input_chunk)

    # Concatenate recorded frames
    RECORDING_BUFFER = numpy.concatenate(recording_frames)[: len(SOUND_DATA)]


def store():
    global sample_id
    sample_id[current_label] += 1
    sound_file = os.path.join(
        DATA_DIR, "{}_{}.wav".format(sample_id[current_label], current_label)
    )
    os.makedirs(os.path.dirname(sound_file), exist_ok=True)
    # Save recording as WAV
    recording_int16 = (RECORDING_BUFFER * 32767).astype(numpy.int16)
    with wave.open(sound_file, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SR)
        wav_file.writeframes(recording_int16.tobytes())


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


def record_acoustic_sample(
    sound_name="long_sweep",
    label="contact",
    sample_id=1,
    base_dir="../../../data/",
    model_name="robot_scan",
    sr=48000,
    channels=1,
    position=None,  # Add position parameter
    metadata=None,  # Add metadata parameter
):
    """
    Standalone function to play an acoustic signal and record the response.
    Saves the recording as a WAV file. No GUI or interactive elements.

    Args:
        sound_name: Name of the sound to play (from SOUNDS dict)
        label: Label for the recording (e.g., "contact", "no_contact")
        sample_id: Unique ID for this sample
        base_dir: Base directory for saving data
        model_name: Model/experiment name for folder structure
        sr: Sample rate (Hz)
        channels: Number of audio channels
        position: Optional tuple (x, y) for robot position
        metadata: Optional dict with additional metadata

    Returns:
        str: Path to the saved WAV file
    """
    import numpy
    import scipy.io.wavfile
    import scipy.signal
    import pyaudio
    import os
    import json

    # PyAudio settings
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paFloat32

    # Define sounds (same as in the main script)
    RECORDING_DELAY_SILENCE = numpy.zeros(int(sr * 0.15), dtype="float32")
    SOUNDS = {
        "sweep": numpy.hstack(
            [
                scipy.signal.chirp(numpy.arange(sr) / sr, 20, 1, 20000).astype(
                    "float32"
                ),
                RECORDING_DELAY_SILENCE,
            ]
        ),
        "long_sweep": numpy.hstack(
            [
                scipy.signal.chirp(numpy.arange(int(sr * 2)) / sr, 20, 2, 20000).astype(
                    "float32"
                ),
                RECORDING_DELAY_SILENCE,
            ]
        ),
        "quick_sweep": numpy.hstack(
            [
                scipy.signal.chirp(
                    numpy.arange(int(sr * 0.3)) / SR, 20, 0.3, 20000
                ).astype("float32"),
                RECORDING_DELAY_SILENCE,
            ]
        ),
        "white_noise": numpy.hstack(
            [
                numpy.random.uniform(low=-0.999, high=1.0, size=(sr)).astype("float32"),
                RECORDING_DELAY_SILENCE,
            ]
        ),
        "silence": numpy.hstack(
            [numpy.zeros((sr,), dtype="float32"), RECORDING_DELAY_SILENCE]
        ),
    }

    # Create data directory
    data_dir = os.path.join(base_dir, model_name, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Get sound data
    if sound_name not in SOUNDS:
        raise ValueError(f"Unknown sound name: {sound_name}")
    sound_data = SOUNDS[sound_name]

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    try:
        # Get default devices
        input_device_index = p.get_default_input_device_info()["index"]
        output_device_index = p.get_default_output_device_info()["index"]

        # Open duplex stream
        stream = p.open(
            format=FORMAT,
            channels=channels,
            rate=sr,
            input=True,
            output=True,
            input_device_index=input_device_index,
            output_device_index=output_device_index,
            frames_per_buffer=CHUNK_SIZE,
        )

        # Play sound and record simultaneously
        recording_frames = []
        num_chunks = len(sound_data) // CHUNK_SIZE
        if len(sound_data) % CHUNK_SIZE != 0:
            num_chunks += 1

        for i in range(num_chunks):
            # Prepare output chunk
            start_idx = i * CHUNK_SIZE
            end_idx = min((i + 1) * CHUNK_SIZE, len(sound_data))
            output_chunk = sound_data[start_idx:end_idx]
            if len(output_chunk) < CHUNK_SIZE:
                output_chunk = numpy.pad(
                    output_chunk, (0, CHUNK_SIZE - len(output_chunk)), "constant"
                )
            output_bytes = output_chunk.tobytes()

            # Write output and read input simultaneously
            stream.write(output_bytes)
            input_bytes = stream.read(CHUNK_SIZE)
            input_chunk = numpy.frombuffer(input_bytes, dtype=numpy.float32)
            recording_frames.append(input_chunk)

        # Close stream
        stream.stop_stream()
        stream.close()

        # Concatenate and trim recording
        recording_buffer = numpy.concatenate(recording_frames)[: len(sound_data)]

        # Save WAV file (simple approach)
        filename = f"{sample_id}_{label}.wav"
        filepath = os.path.join(data_dir, filename)

        # Simple WAV saving without scipy
        import wave
        import struct

        recording_int16 = (recording_buffer * 32767).astype(numpy.int16)
        with wave.open(filepath, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            wav_file.writeframes(recording_int16.tobytes())

        # Save metadata if provided
        if position or metadata:
            metadata_dict = {
                "sample_id": sample_id,
                "label": label,
                "sound_name": sound_name,
                "timestamp": time.time(),
                "sample_rate": sr,
                "channels": channels,
            }
            if position:
                metadata_dict["position"] = {"x": position[0], "y": position[1]}
            if metadata:
                metadata_dict.update(metadata)

            metadata_filename = f"{sample_id}_{label}_metadata.json"
            metadata_filepath = os.path.join(data_dir, metadata_filename)
            with open(metadata_filepath, "w") as f:
                json.dump(metadata_dict, f, indent=2)

        return filepath

    except Exception as e:
        print(f"Recording failed: {e}")
        raise
    finally:
        p.terminate()


def play_sound_only(sound_name="long_sweep", sr=48000):
    """
    Simple function that just plays a sound without recording.
    Perfect for testing or when you only need to send the acoustic signal.

    Args:
        sound_name: Name of the sound to play
        sr: Sample rate (Hz)
    """
    import numpy
    import scipy.signal
    import pyaudio

    # Define sounds
    RECORDING_DELAY_SILENCE = numpy.zeros(int(sr * 0.15), dtype="float32")
    SOUNDS = {
        "sweep": numpy.hstack(
            [
                scipy.signal.chirp(numpy.arange(sr) / sr, 20, 1, 20000).astype(
                    "float32"
                ),
                RECORDING_DELAY_SILENCE,
            ]
        ),
        "long_sweep": numpy.hstack(
            [
                scipy.signal.chirp(numpy.arange(int(sr * 2)) / sr, 20, 2, 20000).astype(
                    "float32"
                ),
                RECORDING_DELAY_SILENCE,
            ]
        ),
        "quick_sweep": numpy.hstack(
            [
                scipy.signal.chirp(
                    numpy.arange(int(sr * 0.3)) / sr, 20, 0.3, 20000
                ).astype("float32"),
                RECORDING_DELAY_SILENCE,
            ]
        ),
        "white_noise": numpy.hstack(
            [
                numpy.random.uniform(low=-0.999, high=1.0, size=(sr)).astype("float32"),
                RECORDING_DELAY_SILENCE,
            ]
        ),
        "silence": numpy.hstack(
            [numpy.zeros((sr,), dtype="float32"), RECORDING_DELAY_SILENCE]
        ),
    }

    if sound_name not in SOUNDS:
        raise ValueError(f"Unknown sound name: {sound_name}")

    sound_data = SOUNDS[sound_name]

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    try:
        # Get default output device
        output_device_index = p.get_default_output_device_info()["index"]

        # Open output stream
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sr,
            output=True,
            output_device_index=output_device_index,
        )

        # Play the sound
        stream.write(sound_data.tobytes())

        # Close stream
        stream.stop_stream()
        stream.close()

        print(f"Played {sound_name} sound")

    except Exception as e:
        print(f"Playback failed: {e}")
        raise
    finally:
        p.terminate()


# Example usage:
# play_sound_only("long_sweep")  # Just play the sound
# filepath = record_acoustic_sample(label="contact", sample_id=1)  # Play and record


if __name__ == "__main__":
    main()
